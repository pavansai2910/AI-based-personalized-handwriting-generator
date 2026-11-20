"""
model.py - inference-only code for the VATr + ResNet-50 handwriting generator.

The model classes are copied EXACTLY from the training notebook (run 3, version 'v6-style')
so that the trained weights in vatr_resnet50_generator.pt load without changes.
No training code is included.
"""
import gzip
import math
import os
import random
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))


# =====================================================================================
# 1. Model architecture (identical to the notebook)
# =====================================================================================
def gray_conv1(conv):
    # 3-channel first convolution -> 1-channel (grayscale) convolution
    c = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
    c.weight.data.copy_(conv.weight.data.sum(1, keepdim=True))
    return c


def sinusoid(n, d):
    # sinusoidal positional encoding table [n, d]
    pos = torch.arange(n).float()[:, None]
    i = torch.arange(0, d, 2).float()
    pe = torch.zeros(n, d)
    div = torch.exp(-math.log(10000.0) * i / d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ResNet50Backbone(nn.Module):
    """ResNet-50 (conv1 .. layer3) adapted to 32-px-high grayscale word images."""
    def __init__(self, d_model):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)    # trained weights come from the checkpoint
        self.stem = nn.Sequential(gray_conv1(r.conv1), r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3 = r.layer1, r.layer2, r.layer3
        self.proj = nn.Conv2d(1024, d_model, 1)

    def features(self, x):
        return self.layer3(self.layer2(self.layer1(self.stem(x))))

    def forward(self, x):
        return self.proj(self.features(x))


class StyleEncoder(nn.Module):
    """15 style images -> 240 style vectors (ResNet-50 + 3-layer Transformer encoder)."""
    def __init__(self, cfg):
        super().__init__()
        self.backbone = ResNet50Backbone(cfg.d_model)
        self.tok_pos = nn.Parameter(torch.randn(256, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.ff_dim, cfg.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.enc_layers, enable_nested_tensor=False)

    def forward(self, style):                       # [B,P,1,32,128] -> [B, P*16, d]
        B, P = style.shape[:2]
        f = self.backbone(style.flatten(0, 1))
        f = f.flatten(2).transpose(1, 2)
        f = f + self.tok_pos[:f.shape[1]]
        return self.encoder(f.reshape(B, -1, f.shape[-1]))


class ResBlockUp(nn.Module):
    def __init__(self, cin, cout, scale=None):
        super().__init__()
        self.scale = scale
        self.bn1, self.c1 = nn.BatchNorm2d(cin), nn.Conv2d(cin, cout, 3, 1, 1)
        self.bn2, self.c2 = nn.BatchNorm2d(cout), nn.Conv2d(cout, cout, 3, 1, 1)
        self.sc = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        h = F.relu(self.bn1(x))
        if self.scale:
            h = F.interpolate(h, scale_factor=self.scale, mode='nearest')
            x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        h = self.c2(F.relu(self.bn2(self.c1(h))))
        return h + self.sc(x)


class Generator(nn.Module):
    """Style encoder + visual-archetype queries + Transformer decoder + CNN image decoder."""
    def __init__(self, cfg, n_writers):
        super().__init__()
        self.cfg = cfg
        self.style_encoder = StyleEncoder(cfg)
        self.style_head = nn.Linear(cfg.d_model, n_writers)     # training only (kept to load the weights)
        self.style_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.query_proj = nn.Linear(256, cfg.d_model)
        self.query_norm = nn.LayerNorm(cfg.d_model)
        self.register_buffer('qpos', sinusoid(128, cfg.d_model), persistent=False)
        layer = nn.TransformerDecoderLayer(cfg.d_model, cfg.n_heads, cfg.ff_dim, cfg.dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.dec_layers)
        self.fc = nn.Linear(cfg.d_model, 512 * 4 * 4)
        self.conv = nn.Sequential(
            ResBlockUp(512, 256, (2, 2)),
            ResBlockUp(256, 128, (2, 2)),
            ResBlockUp(128, 64, (2, 1)),
            ResBlockUp(64, 64, None),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, 3, 1, 1), nn.Tanh())

    def decode(self, S, glyphs, mask, noise_std=0.0):
        # S [N,T,d], glyphs [N,k,256], mask [N,k] -> images [N,1,32,16k] in [-1,1]
        N, k, _ = glyphs.shape
        c = self.query_norm(self.query_proj(glyphs))
        q = c + 0.5 * self.qpos[:k]
        h = self.decoder(q, S, tgt_key_padding_mask=~mask) + c
        h = h + self.style_proj(S.mean(1, keepdim=True))
        if noise_std > 0:
            h = h + noise_std * torch.randn_like(h)
        x = self.fc(h).view(N, k, 512, 4, 4).permute(0, 2, 3, 1, 4).reshape(N, 512, 4, 4 * k)
        img = self.conv(x)
        col = mask.repeat_interleave(self.cfg.char_w, 1)[:, None, None, :]
        return torch.where(col, img, torch.ones_like(img))


class CRNN(nn.Module):
    """Handwriting recognizer R from training (CNN + BiLSTM + CTC). Used here to check readability."""
    def __init__(self, n_classes):
        super().__init__()
        def cbr(i, o):
            return [nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True)]
        self.cnn = nn.Sequential(*cbr(1, 64), nn.MaxPool2d(2), *cbr(64, 128), nn.MaxPool2d(2),
                                 *cbr(128, 256), *cbr(256, 256), nn.MaxPool2d((2, 1)),
                                 *cbr(256, 512), *cbr(512, 512), nn.MaxPool2d((2, 1)),
                                 nn.Conv2d(512, 512, (2, 1)), nn.BatchNorm2d(512), nn.ReLU(True))
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):                       # [N,1,32,W] -> log-probs [T=W/4, N, C]
        f = self.cnn(x).squeeze(2).permute(0, 2, 1).contiguous()
        f, _ = self.rnn(f)
        return self.fc(f).float().log_softmax(-1).permute(1, 0, 2)


# =====================================================================================
# 2. Visual archetypes (GNU Unifont 16x16 glyphs)
# =====================================================================================
def parse_unifont_hex(lines):
    glyphs = {}
    for line in lines:
        line = line.strip()
        if ':' not in line:
            continue
        cp, data = line.split(':', 1)
        try:
            cp = int(cp, 16)
        except ValueError:
            continue
        g = np.zeros((16, 16), np.uint8)
        if len(data) == 32:                 # half-width glyph, centred in 16x16
            for r in range(16):
                bits = int(data[2 * r:2 * r + 2], 16)
                for c in range(8):
                    g[r, 4 + c] = (bits >> (7 - c)) & 1
        elif len(data) == 64:               # full-width glyph
            for r in range(16):
                bits = int(data[4 * r:4 * r + 4], 16)
                for c in range(16):
                    g[r, c] = (bits >> (15 - c)) & 1
        else:
            continue
        glyphs[cp] = g
    return glyphs


class VisualArchetypes:
    def __init__(self, unifont_path=None):
        self.hex, self.cache, self.font = None, {}, None
        if unifont_path and os.path.exists(unifont_path):
            op = gzip.open if unifont_path.endswith('.gz') else open
            with op(unifont_path, 'rt', encoding='utf-8', errors='ignore') as f:
                self.hex = parse_unifont_hex(f)

    def glyph(self, ch):
        if ch in self.cache:
            return self.cache[ch]
        if self.hex is not None and ord(ch) in self.hex:
            g = self.hex[ord(ch)].astype(np.float32)
        else:                                # fallback: render with a default font
            if self.font is None:
                try:
                    self.font = ImageFont.truetype('DejaVuSansMono.ttf', 15)
                except OSError:
                    self.font = ImageFont.load_default()
            im = Image.new('L', (16, 16), 0)
            d = ImageDraw.Draw(im)
            l, t, r, b = d.textbbox((0, 0), ch, font=self.font)
            d.text(((16 - (r - l)) / 2 - l, (16 - (b - t)) / 2 - t), ch, fill=255, font=self.font)
            g = (np.array(im) > 100).astype(np.float32)
        g = g.reshape(-1)
        self.cache[ch] = g
        return g

    def encode(self, texts):
        k = max(len(t) for t in texts)
        G = np.zeros((len(texts), k, 256), np.float32)
        M = np.zeros((len(texts), k), bool)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t):
                G[i, j] = self.glyph(ch)
                M[i, j] = True
        return torch.from_numpy(G), torch.from_numpy(M)


# =====================================================================================
# 3. Loading the trained model
# =====================================================================================
def find_file(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


# All model files (.pt) are kept in DL_PROJECT/results/ and loaded only from there.
RESULTS_DIR = os.environ.get('VATR_RESULTS_DIR', os.path.normpath(os.path.join(HERE, '..', 'results')))


def default_model_path():
    return find_file([os.path.join(RESULTS_DIR, 'vatr_resnet50_generator_avg.pt'),
                      os.path.join(RESULTS_DIR, 'vatr_resnet50_generator.pt'),
                      os.path.join(RESULTS_DIR, 'checkpoints', 'best_G.pt')])


def available_models():
    """generator weight files in results/, best choice first: (label, path)"""
    cands = [('Averaged: step 24k + 45.9k (recommended)', os.path.join(RESULTS_DIR, 'vatr_resnet50_generator_avg.pt')),
             ('Best checkpoint: step 24k', os.path.join(RESULTS_DIR, 'vatr_resnet50_generator.pt'))]
    return [(label, p) for label, p in cands if os.path.exists(p)]


def default_recognizer_path():
    return find_file([os.path.join(RESULTS_DIR, 'recognizer.pt')])


def default_unifont_path():
    return find_file([os.path.join(RESULTS_DIR, 'cache', 'unifont.hex.gz')])


class HandwritingModel:
    """Loads the exported generator once and writes text in a given style."""
    def __init__(self, model_path=None, unifont_path=None, device=None):
        model_path = model_path or default_model_path()
        if model_path is None:
            raise FileNotFoundError(f'No generator weights found in {RESULTS_DIR} '
                                    '(expected vatr_resnet50_generator_avg.pt or vatr_resnet50_generator.pt).')
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        ck = torch.load(model_path, map_location='cpu', weights_only=False)
        state = ck['G']
        self.cfg = SimpleNamespace(**ck['cfg'])
        self.charset = ck.get('charset', '')
        self.info = {k: ck.get(k) for k in ('step', 'best_step', 'best_validation') if k in ck}
        n_writers = state['style_head.weight'].shape[0]
        self.G = Generator(self.cfg, n_writers)
        self.G.load_state_dict(state, strict=True)
        self.G.to(self.device).eval()
        self.arch = VisualArchetypes(unifont_path or default_unifont_path())
        self.unifont_loaded = self.arch.hex is not None
        self.model_path = model_path
        # optional: the recognizer R (for "best of N" readability selection)
        self.R, rpath = None, default_recognizer_path()
        if rpath:
            rk = torch.load(rpath, map_location='cpu', weights_only=False)
            self.r_charset = rk['charset']
            self.R = CRNN(len(self.r_charset) + 1)
            self.R.load_state_dict(rk['R'], strict=True)
            self.R.to(self.device).eval()
            self.r_index = {c: i + 1 for i, c in enumerate(self.r_charset)}

    # ---------------- style preparation ----------------
    def style_format(self, a):
        # uint8 32xW -> 32x128 (squeeze if wider, white padding if narrower)
        W = self.cfg.style_w
        h, w = a.shape
        if w > W:
            return cv2.resize(a, (W, h), interpolation=cv2.INTER_AREA)
        out = np.full((h, W), 255, np.uint8)
        out[:, :w] = a
        return out

    def to_height(self, a):
        # resize a grayscale word crop to 32 px height (width 4..384), like the training data
        h, w = a.shape
        nw = int(np.clip(round(w * self.cfg.img_h / h), 4, self.cfg.max_real_w))
        return cv2.resize(a, (nw, self.cfg.img_h), interpolation=cv2.INTER_AREA)

    def style_tensor(self, crops, seed=0):
        """list of uint8 grayscale word crops -> ([15,1,32,128] tensor, the 15 crops used)"""
        crops = [c for c in crops if c is not None and c.size > 0 and min(c.shape) >= 2]
        if not crops:
            raise ValueError('No handwriting found in the style images.')
        P = self.cfg.num_style
        rng = random.Random(seed)
        crops = sorted(crops, key=lambda c: -c.shape[1] / max(1, c.shape[0]))[:max(P, len(crops) // 2)]
        pick = rng.sample(crops, P) if len(crops) >= P else rng.choices(crops, k=P)
        imgs = []
        for c in pick:
            r = self.style_format(self.to_height(c))
            imgs.append(torch.from_numpy(r).float().div(127.5).sub(1.0))
        return torch.stack(imgs)[:, None], pick

    # ---------------- generation ----------------
    @torch.no_grad()
    def generate_words(self, style, words, noise_std=0.0, chunk=64):
        S = self.G.style_encoder(style[None].to(self.device))
        out = []
        for i in range(0, len(words), chunk):
            ws = words[i:i + chunk]
            g, m = self.arch.encode(ws)
            imgs = self.G.decode(S.expand(len(ws), -1, -1), g.to(self.device), m.to(self.device), noise_std)
            for j, w in enumerate(ws):
                out.append(imgs[j, 0, :, :len(w) * self.cfg.char_w].float().cpu().numpy())
        return out

    # ---------------- automatic "match the handwriting" settings ----------------
    def style_profile(self, style, crops, texts=None, seed=0,
                      probe='handwriting generator quickly minimum letters'):
        """Measures the writer's slant, letter width and stroke thickness from the style samples and
        compares them with what the generator produces, so that the post-processing can match them.
        texts: what each crop says (exact letter width); without it the recognizer estimates it."""
        torch.manual_seed(seed)
        words = probe.split()
        gen = [equalize_chars(g, len(w), self.cfg.char_w) for g, w in zip(self.generate_words(style, words, self.cfg.noise_std), words)]
        crops32 = [self.to_height(c) for c in crops if c is not None and c.size > 0]
        # slant
        s_style = float(np.median([estimate_slant(c) for c in crops32]))
        s_gen = float(np.median([estimate_slant(g) for g in gen]))
        # letter width
        gen_cw = float(np.median([trim_word(np.where(g > 0.55, 1.0, g)).shape[1] / len(w) for g, w in zip(gen, words)]))
        real_cw, source = None, None
        if texts and len(texts) == len(crops32):
            real_cw, source = float(np.median([c.shape[1] / max(1, len(t)) for c, t in zip(crops32, texts)])), 'exact'
        elif self.R is not None:
            reads = self.read([c.astype(np.float32) / 127.5 - 1 for c in crops32])
            ok = [(c, r) for c, r in zip(crops32, reads) if len(r) >= 2]
            if ok:
                real_cw, source = float(np.median([c.shape[1] / len(r) for c, r in ok])), 'estimated'
        squeeze = float(np.clip(real_cw / gen_cw, 0.55, 1.6)) if real_cw else 0.8
        # stroke thickness
        sw_style = float(np.median([stroke_width(c) for c in crops32]))
        sw_gen = float(np.median([stroke_width(g) for g in gen]))
        return dict(slant_shift=float(np.clip(s_gen - s_style, -0.6, 0.6)), squeeze=squeeze,
                    width_source=source or 'default', real_char_w=real_cw, gen_char_w=gen_cw,
                    stroke_style=sw_style, stroke_gen=sw_gen, ink=ink_level(crops))

    # ---------------- readability (uses the recognizer R) ----------------
    def _batch(self, imgs):
        W = max(16, int(np.ceil(max(im.shape[1] for im in imgs) / 16) * 16))
        x = torch.ones(len(imgs), 1, self.cfg.img_h, W)
        for i, im in enumerate(imgs):
            x[i, 0, :, :im.shape[1]] = torch.from_numpy(np.clip(im, -1, 1))
        return x.to(self.device), torch.tensor([im.shape[1] for im in imgs])

    @torch.no_grad()
    def read(self, imgs):
        """recognise word images (float [-1,1], 32 px high) -> strings"""
        if self.R is None or not imgs:
            return [None] * len(imgs)
        x, widths = self._batch(imgs)
        best = self.R(x).argmax(-1).t().cpu().numpy()
        out = []
        for n, seq in enumerate(best):
            seq = seq[:max(1, int(widths[n]) // 4)]
            chars, prev = [], 0
            for c in seq:
                if c != prev and c != 0:
                    chars.append(self.r_charset[c - 1])
                prev = c
            out.append(''.join(chars))
        return out

    @torch.no_grad()
    def readability_cost(self, imgs, texts):
        """CTC loss per character of the intended text (lower = easier to read as that text)"""
        x, widths = self._batch(imgs)
        lp = self.R(x)
        tg = [[self.r_index[c] for c in t if c in self.r_index] for t in texts]
        tl = torch.tensor([max(1, len(t)) for t in tg])
        flat = torch.tensor([c for t in tg for c in (t if t else [1])])
        il = torch.clamp(widths // 4, 1, lp.shape[0])
        loss = F.ctc_loss(lp, flat, il, tl, blank=0, reduction='none', zero_infinity=True)
        return (loss / tl).cpu().numpy()

    def generate_words_best(self, style, words, noise_std=0.2, n_candidates=6, seed=0):
        """write every word n_candidates times (different stroke noise) and keep the version the
        recognizer reads most confidently as the intended word -> much more legible text."""
        if self.R is None or n_candidates <= 1:
            torch.manual_seed(seed)
            return self.generate_words(style, words, noise_std)
        best, best_cost = None, None
        for i in range(n_candidates):
            torch.manual_seed(seed * 1000 + i)
            ns = noise_std if i == 0 else max(noise_std, 0.3)          # candidate 0 = normal setting
            cand = self.generate_words(style, words, ns)
            cost = self.readability_cost(cand, words)
            if best is None:
                best, best_cost = cand, cost
            else:
                for j in range(len(words)):
                    if cost[j] < best_cost[j]:
                        best[j], best_cost[j] = cand[j], cost[j]
        return best

    def write_text(self, style, text, max_width=1000, noise_std=None, seed=0, clean=True, word_gap=0.45,
                   squeeze=1.0, slant_shift=0.0, clean_punct=False, n_candidates=1, equalize=True):
        """text -> page image (float in [-1,1], white = 1).
        Optional post-processing (no retraining):
          squeeze      horizontal compression of every word (1.0 = as generated, 0.7 = 30 % narrower)
          slant_shift  extra shear applied to every word (computed by match_slant_shift)
          clean_punct  '.' and ',' at the end of a word are drawn as clean marks instead of generated
        """
        if noise_std is None:
            noise_std = self.cfg.noise_std
        torch.manual_seed(seed)
        tokens = []
        for li, line in enumerate(text.split('\n')):
            if li > 0:
                tokens.append('\n')
            tokens += line.split()
        # split trailing '.'/',' off the words that are generated
        cores, puncts = [], []
        for t in tokens:
            if t == '\n':
                continue
            core, p = t, ''
            if clean_punct:
                while core and core[-1] in '.,':
                    p = core[-1] + p
                    core = core[:-1]
            cores.append(core)
            puncts.append(p)
        if not cores:
            return np.ones((self.cfg.img_h, 64), np.float32)
        to_gen = [c for c in cores if c]
        gen_imgs = self.generate_words_best(style, to_gen, noise_std, n_candidates, seed) if to_gen else []
        self.last_words = list(zip(to_gen, gen_imgs))          # raw images, for the readability check
        it = iter(gen_imgs)
        word_imgs, k = [], 0
        for t in tokens:
            if t == '\n':
                word_imgs.append(None)
                continue
            core, p = cores[k], puncts[k]
            k += 1
            im = next(it).copy() if core else np.ones((self.cfg.img_h, 2), np.float32)
            if core and equalize:
                im = equalize_chars(im, len(core), self.cfg.char_w)
            if clean:
                im[im > 0.55] = 1.0                       # remove the faint grey haze around letters
            if core:
                im = trim_word(im, pad=1)
                if squeeze != 1.0:
                    im = cv2.resize(im, (max(4, int(round(im.shape[1] * squeeze))), im.shape[0]),
                                    interpolation=cv2.INTER_AREA)
                if abs(slant_shift) > 1e-3:
                    im = shear_word(im, slant_shift)
            if p:
                im = append_punct(im, p)
            word_imgs.append(im)
        return compose_lines(word_imgs, tokens, max_width, int(self.cfg.img_h * word_gap))

    # ---------------- style statistics used by the post-processing ----------------
    def match_slant_shift(self, style, style_crops, seed=0, probe='handwriting generator quickly'):
        """shear that gives generated words the same slant as the style samples"""
        s_style = np.median([estimate_slant(c) for c in style_crops])
        torch.manual_seed(seed)
        probes = self.generate_words(style, probe.split(), self.cfg.noise_std)
        s_gen = np.median([estimate_slant(((np.clip(p, -1, 1) + 1) * 127.5).astype(np.uint8)) for p in probes])
        return float(np.clip(s_gen - s_style, -0.6, 0.6)), float(s_style), float(s_gen)


# =====================================================================================
# 3b. Post-processing helpers (style-aware, no retraining)
# =====================================================================================
def equalize_chars(im, n_chars, char_w=16, target=0.9, max_gain=5.0, floor=0.06):
    """Per-character contrast boost: the generator sometimes draws a letter (often a final 'e')
    so faintly that people cannot see it. Every 16-px character slot is rescaled so that its
    darkest strokes reach the same ink strength; gains are interpolated between slots (no seams)."""
    d = (1 - np.clip(im, -1, 1)) / 2                       # darkness 0 (white) .. 1 (black)
    gains = []
    for j in range(n_chars):
        blk = d[:, j * char_w:(j + 1) * char_w]
        core = float(np.percentile(blk, 97)) if blk.size else 0
        gains.append(1.0 if core < floor else float(np.clip(target / core, 1.0, max_gain)))
    if not gains:
        return im
    centers = (np.arange(n_chars) + 0.5) * char_w
    col_gain = np.interp(np.arange(d.shape[1]), centers, gains).astype(np.float32)
    d = np.clip(d * col_gain[None, :], 0, 1)
    return 1 - 2 * d


def _shear(img, s, fill):
    # x' = x + s * (y - h/2); output width grows so nothing is cut off
    h, w = img.shape[:2]
    pad = int(np.ceil(abs(s) * h / 2)) + 1
    M = np.float32([[1, s, pad - s * h / 2], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w + 2 * pad, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def estimate_slant(gray):
    """slant of a word image (uint8, white background): the shear that makes the strokes
    most vertical (maximises the variance of the column ink profile)."""
    g = gray if gray.dtype == np.uint8 else ((np.clip(gray, -1, 1) + 1) * 127.5).astype(np.uint8)
    _, bw = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if bw.sum() < 10:
        return 0.0
    bw = bw.astype(np.float32)
    best, best_s = -1, 0.0
    for s in np.linspace(-1.0, 1.0, 41):
        col = _shear(bw, s, 0).sum(0)
        score = float((col ** 2).sum())
        if score > best:
            best, best_s = score, s
    return best_s


def stroke_width(img):
    """typical pen-stroke thickness in pixels of a 32-px-high word image (uint8 or float [-1,1])"""
    g = img if img.dtype == np.uint8 else ((np.clip(img, -1, 1) + 1) * 127.5).astype(np.uint8)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if bw.sum() == 0:
        return 0.0
    dt = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    ridge = dt[(dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (bw > 0)]
    return 2.0 * float(np.median(ridge)) if ridge.size else 0.0


def shear_word(img, s):
    """apply the shear s to a generated word (float [-1,1], white = 1) and trim it again"""
    return trim_word(_shear(img, s, 1.0), pad=1)


def ink_level(crops):
    """typical (dark) ink tone of the style samples, 0 = black ... 1 = white"""
    vals = []
    for c in crops:
        _, bw = cv2.threshold(c, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if bw.sum() > 0:
            vals.append(np.percentile(c[bw > 0], 25) / 255.0)    # core of the strokes, not the grey edges
    return float(np.clip(np.median(vals) if vals else 0.15, 0.0, 0.35))


def append_punct(im, p, H=32):
    """draw clean '.' / ',' marks after a word (float image, white = 1)"""
    ink = im[im < 0.3]
    level = float(np.median(ink)) if ink.size else -0.8
    for ch in p:
        mark = np.ones((H, 8), np.float32)
        cv2.circle(mark, (3, 26), 2, level, -1, lineType=cv2.LINE_AA)
        if ch == ',':
            cv2.line(mark, (4, 27), (2, 31), level, 1, lineType=cv2.LINE_AA)
        im = np.concatenate([im, mark], 1)
    return im


def best_weight(page, target_stroke, scale=2, ink=0.15, candidates=(-1, 0, 1, 2)):
    """stroke-thickness setting whose rendered strokes are closest to the writer's real stroke width"""
    if not target_stroke:
        return 0
    errs = []
    for w in candidates:
        img = np.array(render_page(page, scale, True, ink, w))
        errs.append(abs(stroke_width(img) / scale - target_stroke))
    return int(candidates[int(np.argmin(errs))])


def render_page(page, scale=2, crisp=True, ink=0.15, weight=0, color=None):
    """page (float [-1,1]) -> PIL image.
    crisp=True : smooth up-scaling + contrast normalisation + soft threshold -> sharp, anti-aliased
                 strokes instead of blurry grey ones (adapts to how dark the generated strokes are).
    ink        : ink tone 0 = black ... 1 = white (used when color is None)
    weight     : stroke thickness adjustment in output pixels (-1 thinner ... +3 bolder)
    color      : optional RGB ink colour, e.g. (20, 40, 140) for a blue pen."""
    v = (np.clip(page, -1, 1) + 1) / 2                      # 0 = black, 1 = white
    if scale != 1:
        v = cv2.resize(v, (v.shape[1] * scale, v.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
    v = np.clip(v, 0, 1).astype(np.float32)
    if not crisp:
        return Image.fromarray((v * 255).astype(np.uint8))
    v = cv2.GaussianBlur(v, (0, 0), 0.5 * scale)
    d = 1 - v                                               # darkness
    strokes = d[d > 0.25]
    core = float(np.percentile(strokes, 90)) if strokes.size else 0.7
    lo = 0.08
    hi = max(lo + 0.12, 0.5 * core)
    alpha = np.clip((d - lo) / (hi - lo), 0, 1)
    if weight:
        k = int(abs(weight)) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        alpha = cv2.dilate(alpha, kernel) if weight > 0 else cv2.erode(alpha, kernel)
    if color is None:
        out = 1 - alpha * (1 - ink)
        return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    rgb = np.ones(alpha.shape + (3,), np.float32)
    for ch in range(3):
        rgb[..., ch] = 1 - alpha * (1 - color[ch] / 255.0)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), 'RGB')


# =====================================================================================
# 4. Page layout helpers
# =====================================================================================
def trim_word(img, thr=0.3, pad=2):
    ink = (img < thr).any(0)
    if not ink.any():
        return img[:, :8]
    l, r = np.argmax(ink), len(ink) - np.argmax(ink[::-1])
    return img[:, max(0, l - pad):min(img.shape[1], r + pad)]


def compose_lines(word_imgs, tokens, max_width=1000, gap=14, line_gap=6, H=32):
    lines, cur, cur_w = [], [], 0
    for im, t in zip(word_imgs, tokens):
        if t == '\n':
            lines.append(cur)
            cur, cur_w = [], 0
            continue
        im = trim_word(im)
        if cur and cur_w + gap + im.shape[1] > max_width:
            lines.append(cur)
            cur, cur_w = [], 0
        cur.append(im)
        cur_w += im.shape[1] + (gap if len(cur) > 1 else 0)
    lines.append(cur)
    rows = []
    for ln in lines:
        row = np.ones((H, max_width), np.float32)
        x = 0
        for im in ln:
            w = min(im.shape[1], max_width - x)
            if w <= 0:
                break
            row[:, x:x + w] = im[:, :w]
            x += w + gap
        rows += [row, np.ones((line_gap, max_width), np.float32)]
    page = np.concatenate(rows, 0)
    used = np.where((page < 0.3).any(0))[0]
    return page[:, :used.max() + 10] if len(used) else page


def to_pil(page, scale=2):
    img = Image.fromarray(((np.clip(page, -1, 1) + 1) * 127.5).astype(np.uint8))
    return img.resize((img.width * scale, img.height * scale), Image.BICUBIC) if scale != 1 else img


# =====================================================================================
# 5. Using your own handwriting (photo of a page / word images)
# =====================================================================================
def normalize_page(gray):
    # remove uneven illumination, stretch contrast -> white paper, dark ink
    k = max(31, (min(gray.shape) // 20) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    norm = cv2.divide(gray, bg, scale=255)
    lo = np.percentile(norm, 1)
    return np.clip((norm.astype(np.float32) - lo) / max(1, 255 - lo) * 255, 0, 255).astype(np.uint8)


def segment_words(gray):
    """photo of a handwritten page -> (word crops in reading order, preview image with boxes)"""
    if max(gray.shape) > 2500:
        s = 2500 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    norm = normalize_page(gray)
    _, bw = cv2.threshold(cv2.GaussianBlur(norm, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT][stats[1:, cv2.CC_STAT_AREA] > 15]
    if len(hs) == 0:
        return [], cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    xh = float(np.median(hs))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(xh * 0.5)), max(1, int(xh * 0.35))))
    merged = cv2.dilate(bw, kernel)
    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cnts]
    boxes = [b for b in boxes if b[3] > 0.6 * xh and b[2] > 0.5 * xh and b[3] < 8 * xh]
    boxes.sort(key=lambda b: b[1] + b[3] / 2)
    lines, cur = [], []
    for b in boxes:
        if cur and abs((b[1] + b[3] / 2) - np.mean([c[1] + c[3] / 2 for c in cur])) > 1.2 * xh:
            lines.append(cur)
            cur = []
        cur.append(b)
    if cur:
        lines.append(cur)
    crops, vis = [], cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    for ln in lines:
        for x, y, w, h in sorted(ln, key=lambda b: b[0]):
            m = bw[y:y + h, x:x + w] > 0
            ys, xs = np.where(m)
            if len(ys) == 0:
                continue
            y0, y1, x0, x1 = y + ys.min(), y + ys.max() + 1, x + xs.min(), x + xs.max() + 1
            crops.append(norm[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + 3])
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return crops, vis


def read_gray(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
