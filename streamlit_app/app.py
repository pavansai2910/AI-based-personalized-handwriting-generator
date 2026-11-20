"""
app.py - Streamlit web interface for the personalized handwriting generator.

Run:   streamlit run app.py
"""
import io
import os
import time

import cv2
import numpy as np
import streamlit as st

import importlib
import json

import model as model_module

# Streamlit keeps imported modules in memory while the app runs. If model.py was changed (e.g. updated),
# reload it so the new code is used without having to restart the server.
_mtime = os.path.getmtime(model_module.__file__)
if getattr(model_module, '_LOADED_MTIME', _mtime) != _mtime:
    model_module = importlib.reload(model_module)
    st.cache_resource.clear()
model_module._LOADED_MTIME = _mtime

from model import (HandwritingModel, available_models, best_weight, ink_level, read_gray, render_page,
                   segment_words, to_pil)

INK_COLORS = {"Writer's ink tone": None, 'Black pen': (25, 25, 30), 'Blue pen': (20, 45, 140),
              'Dark grey pencil': (70, 70, 75)}

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(HERE, 'samples')

st.set_page_config(page_title='Handwriting Generator', layout='wide')


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner='Loading the trained model ...')
def load_model(path=None, code_version=None):
    # code_version (modification time of model.py) is part of the cache key: when model.py changes,
    # a fresh model object is created instead of re-using one built from the old code
    return HandwritingModel(model_path=path)


def model_code_version():
    import model as model_module
    return os.path.getmtime(model_module.__file__)


@st.cache_data
def list_sample_writers():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    return sorted(d for d in os.listdir(SAMPLES_DIR) if os.path.isdir(os.path.join(SAMPLES_DIR, d)))


def load_sample_writer(name):
    """crops + what each word says (texts.json, from the IAM transcriptions) for exact letter widths"""
    folder = os.path.join(SAMPLES_DIR, name)
    try:
        transcripts = json.load(open(os.path.join(folder, 'texts.json'), encoding='utf-8'))
    except (OSError, ValueError):
        transcripts = {}
    crops, texts = [], []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                crops.append(img)
                texts.append(transcripts.get(f))
    return crops, (texts if all(texts) else None)


def show_image(container, img, caption=None):
    """st.image that fills its column on ANY Streamlit version
    (newer versions use width='stretch', 1.40-1.4x use_container_width, older use_column_width)."""
    for kw in ({'width': 'stretch'}, {'use_container_width': True}, {'use_column_width': True}, {}):
        try:
            return container.image(img, caption=caption, **kw)
        except Exception:
            continue


def show_style_preview(crops, n=15):
    st.caption(f'Style samples used ({min(n, len(crops))} words)')
    cols = st.columns(5)
    for i, c in enumerate(crops[:n]):
        show_image(cols[i % 5], c)


# ------------------------------------------------------------------ page
st.title('Personalized Handwriting Generator')
st.write('Type any text, choose a handwriting style, and the model writes your text in that style. '
         'Model: VATr (CVPR 2023) with a ResNet-50 style encoder, trained on the IAM dataset.')

MODELS = available_models()
with st.sidebar:
    if len(MODELS) > 1:
        label = st.selectbox('Generator weights', [l for l, _ in MODELS], index=0,
                             help='Averaged = mean of the best (step 24k) and final (step 45.9k) generator weights '
                                  '("model soup"): fewer stray marks, slightly stronger style.')
        model_path = dict(MODELS)[label]
    else:
        model_path = MODELS[0][1] if MODELS else None
try:
    model = load_model(model_path, model_code_version())
    if not hasattr(model, 'style_profile'):          # safety net: stale object from older code
        st.cache_resource.clear()
        model = load_model(model_path, model_code_version())
except Exception as e:
    st.error(f'Could not load the model: {e}')
    st.stop()

# ---------------- sidebar: style + settings
with st.sidebar:
    st.header('1. Handwriting style')
    source = st.radio('Style source',
                      ['IAM writer (unseen during training)', 'Photo of a handwritten page', 'Word images'],
                      index=0)
    crops, preview, crop_texts = None, None, None

    if source.startswith('IAM'):
        writers = list_sample_writers()
        if not writers:
            st.warning('No sample writers found in the samples/ folder.')
        else:
            name = st.selectbox('Writer', writers, format_func=lambda d: d.replace('writer_', 'Writer '))
            crops, crop_texts = load_sample_writer(name)

    elif source.startswith('Photo'):
        up = st.file_uploader('Upload a photo / scan (JPG or PNG)', type=['png', 'jpg', 'jpeg'])
        st.caption('Tip: about 20+ words, dark pen on white paper, photographed flat in good light.')
        if up is not None:
            gray = read_gray(up.getvalue())
            if gray is None:
                st.error('Could not read the image.')
            else:
                crops, preview = segment_words(gray)
                st.success(f'{len(crops)} words found')

    else:
        ups = st.file_uploader('Upload single-word images', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        if ups:
            crops = [read_gray(u.getvalue()) for u in ups]
            crops = [c for c in crops if c is not None]

    st.header('2. Settings')
    seed = st.number_input('Random seed (variation)', min_value=0, max_value=99999, value=0, step=1)
    noise = st.slider('Stroke variation (noise)', 0.0, 0.5, float(model.cfg.noise_std), 0.05)
    width = st.slider('Line width (pixels)', 400, 1600, 1000, 50)
    clean = st.checkbox('Clean background (remove grey haze)', value=True)
    scale = st.select_slider('Output zoom', options=[1, 2, 3], value=2)

    st.header('3. Appearance (post-processing)')
    preset = st.radio('Preset', ['Match the handwriting (recommended)', 'Most readable', 'Custom'], index=0,
                      help='Match: letter width, stroke thickness, slant and ink tone are measured from the style '
                           'samples and applied automatically. Most readable: medium spacing, black ink. '
                           'Custom: set everything yourself.')
    # defaults used by the presets; the controls below are only shown for "Custom"
    equalize, enhance, match_slant, clean_punct = True, True, True, True
    squeeze, weight, ink_choice = 0.8, 0, "Writer's ink tone"
    if preset == 'Most readable':
        ink_choice = 'Black pen'
    elif preset == 'Custom':
        equalize = st.checkbox('Boost faint letters (more readable)', value=True,
                               help='The model sometimes draws a letter (often a final e) very faintly. '
                                    'Each character is brought to the same ink strength.')
        enhance = st.checkbox('Sharp ink rendering', value=True,
                              help='Smooth up-scaling + adaptive thresholding: crisp strokes instead of blurry grey.')
        squeeze = st.slider('Letter spacing (word width)', 0.55, 1.6, 0.8, 0.05,
                            help='The model draws every character 16 px wide. Lower = narrower letters.')
        match_slant = st.checkbox("Match the writer's slant", value=True)
        clean_punct = st.checkbox('Clean full stops and commas', value=True)
        weight = st.slider('Stroke thickness', -1, 3, 0, 1, disabled=not enhance)
        ink_choice = st.selectbox('Ink colour', list(INK_COLORS), index=0, disabled=not enhance)
    if model.R is not None:
        n_cand = st.select_slider('Versions tried per word (keep the most readable)', options=[1, 2, 4, 6], value=1,
                                  help='Each word is written several times; the recognizer from training keeps '
                                       'the clearest version. Slower: time grows with the number of versions.')
    else:
        n_cand = 1
    show_raw = st.checkbox('Also show the raw model output (for comparison)', value=False)

    with st.expander('Model info'):
        st.write(f'Device: **{model.device}**')
        st.write(f'Weights: `{os.path.relpath(model.model_path, HERE)}`')
        st.write('Unifont glyphs: ' + ('loaded' if model.unifont_loaded else 'fallback font'))
        st.write('Run 3 evaluation (best checkpoint, step 24,000): CER 0.0055 | FID 51.1 | '
                 'writer-ID 50.9 % | unseen-writer retrieval 71.1 %')

# ---------------- main area
left, right = st.columns([1, 1])
with left:
    text = st.text_area('Text to write', height=150,
                        value='Hello! This is my personalized handwriting generator.\n'
                              'It writes any text in the style you choose.')
    unknown = sorted(set(c for c in text if c not in model.charset and not c.isspace()))
    if unknown:
        st.info('Characters not seen in training (quality may be lower): ' + ' '.join(unknown))
    go = st.button('Generate handwriting', type='primary')

with right:
    if preview is not None:
        show_image(st, preview, caption='Detected words (red boxes)')
    if crops:
        try:
            style, used = model.style_tensor(crops, seed=0)
            show_style_preview(used)
        except ValueError as e:
            st.error(str(e))
            crops = None
    else:
        st.info('Choose a handwriting style in the sidebar.')

st.divider()
if go:
    if not crops:
        st.warning('Please choose or upload a handwriting style first.')
    elif not text.strip():
        st.warning('Please type some text.')
    else:
        with st.spinner('Writing ...'):
            t0 = time.time()
            profile = model.style_profile(style, crops, crop_texts, seed=int(seed))
            shift = profile['slant_shift'] if match_slant else 0.0
            if preset.startswith('Match'):
                squeeze = profile['squeeze']
            page = model.write_text(style, text, max_width=width, noise_std=noise, seed=int(seed), clean=clean,
                                    squeeze=squeeze, slant_shift=shift, clean_punct=clean_punct,
                                    n_candidates=n_cand, equalize=equalize)
            words_raw = list(model.last_words)
            if preset.startswith('Match'):
                weight = best_weight(page, profile['stroke_style'], scale, profile['ink'])
            if enhance:
                img = render_page(page, scale=scale, crisp=True, ink=profile['ink'], weight=weight,
                                  color=INK_COLORS[ink_choice])
            else:
                img = to_pil(page, scale=scale)
        st.subheader('Result')
        st.image(img)
        if show_raw:
            raw = to_pil(model.write_text(style, text, max_width=width, noise_std=noise, seed=int(seed),
                                          clean=False), scale=scale)
            st.caption('Raw model output (no post-processing)')
            st.image(raw)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        st.download_button('Download PNG', buf.getvalue(), file_name='handwriting.png', mime='image/png')
        st.caption(f'Generated {len(text.split())} words in {time.time() - t0:.1f} s on {model.device}.')
        if preset.startswith('Match'):
            src = {'exact': 'measured exactly', 'estimated': 'estimated by the recognizer',
                   'default': 'default'}[profile['width_source']]
            st.caption(f"Matched to the handwriting: letter width x{squeeze:.2f} ({src}), "
                       f"stroke thickness {weight:+d}, slant correction {shift:+.2f}. "
                       "Choose 'Custom' in the sidebar to adjust.")
        if model.R is not None and words_raw:
            with st.expander('Readability check (what the recognizer from training reads)'):
                reads = model.read([im for _, im in words_raw])
                ok = sum(a == b for (a, _), b in zip(words_raw, reads))
                st.write(f'{ok} of {len(reads)} words read exactly.')
                bad = [f'"{a}" -> "{b}"' for (a, _), b in zip(words_raw, reads) if a != b]
                if bad:
                    st.write('Misread: ' + ', '.join(bad))
                st.caption('Note: this recognizer guided training, so it reads the model\'s writing more easily '
                           'than a person does.')
