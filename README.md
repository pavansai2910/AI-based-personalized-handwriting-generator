# Personalized Handwriting Generator (VATr + ResNet-50)

Generates an image of any typed text in the handwriting style of a given writer, from only
15 word images of that writer. The model is an implementation of **VATr** (Visual
Archetypes-based Transformer; V. Pippi, S. Cascianelli, R. Cucchiara, *Handwritten Text
Generation from Visual Archetypes*, CVPR 2023) with a **ResNet-50** style encoder, trained on
the **IAM Handwriting Database**. A Streamlit web interface is included.

| Item | Value |
|---|---|
| Training notebook | `dl-final-project.ipynb` (Kaggle, 1 x Tesla T4, ~11 h) |
| Main file to run | `streamlit_app/app.py` |
| Final model | Run 3, best checkpoint at step 24,000 |
| Key results (unseen writers) | CER 0.0055, WER 0.0218, FID 51.1, GS 0.0094, retrieval 71.1 % |

---

## 1. Requirements

**Python:** 3.10, 3.11 or 3.12 (64-bit). Training was run on Kaggle with Python 3,
PyTorch 2.11 (CUDA) and torchvision 0.26.

**Hardware:**
- Running the web app: CPU is enough (a few seconds per sentence); a GPU is used automatically if available.
- Training: an NVIDIA GPU with at least 15 GB memory (Kaggle T4 / P100 recommended).

**Libraries** (all listed in `requirements.txt`):

| Library | Used for |
|---|---|
| torch, torchvision | models, training, ResNet-50 / ResNet-18 / Inception-v3 |
| numpy, opencv-python, pillow | image loading, processing, page segmentation |
| streamlit | web interface |
| matplotlib | plots and sample sheets |
| scipy, scikit-learn, pandas | metrics (precision/recall/F1, kappa, MCC, confusion matrices), tables |
| gudhi | Geometry Score (persistent homology) |
| scikit-image | SSIM evaluation |
| kagglehub, jupyter | optional: dataset download, running the notebook locally |

`streamlit_app/requirements.txt` contains only the six packages needed to run the app.

---

## 2. Installation

```bash
# 1. go to the project folder
cd DL_PROJECT

# 2. create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # Linux / macOS

# 3. install the dependencies
pip install --upgrade pip
pip install -r requirements.txt                    # everything (training + app)
# or, only for the web app:
pip install -r streamlit_app/requirements.txt
```

For GPU training on your own machine, install the CUDA build of PyTorch first, following
https://pytorch.org/get-started/locally/ (for example
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`).

Check the installation:

```bash
python -c "import torch, torchvision, cv2, streamlit; print(torch.__version__, torch.cuda.is_available())"
```

---

## 3. Dataset

The project uses the word-level part of the **IAM Handwriting Database** (657 writers,
95,927 usable words after filtering).

**official source:** register and download from
https://fki.tic.heia-fr.ch/databases/iam-handwriting-database (files `words.tgz`,
`ascii.tgz`; `forms.txt` is inside the ascii archive).

**Placement for local use** - put the files in `dataset/` exactly like this:

```
dataset/
  ascii/
    words.txt          word id, segmentation status, transcription
    forms.txt          form id -> writer id
  words/
    a01/
      a01-000u/
        a01-000u-00-00.png
        ...
```

The notebook searches the input folder recursively for `words.txt`, `forms.txt` and the
`words/` images, so an extra top-level folder (for example `dataset/iam/...`) also works.
The dataset is not stored in git (`.gitignore`). The web app does **not** need the dataset:
it ships with sample words of 20 unseen IAM writers in `streamlit_app/samples/`.

---

## 4. Training

### On Kaggle (recommended)
1. Create a new notebook and upload `dl-final-project.ipynb` (*File -> Import Notebook*).
2. *Add Input* -> dataset `venkatanarayana55/dataset-dl`.
3. *Settings*: Accelerator **GPU T4 x1** (or P100), Internet **on**.
4. *Save Version* -> **Save & Run All (Commit)**.

The notebook runs end to end: data preparation, stage 1 (ResNet-50, recognizer and writer
classifier pre-training, 3 epochs), stage 2 (VATr GAN training, time budget 11 h, validation
every 4,000 steps, best model kept automatically), then evaluation, sample generation and
export. All outputs are written to `/kaggle/working`. If a session ends early, run it again
with the previous output attached: training resumes from `checkpoints/last.pt`.

### Locally
```bash
# Windows (PowerShell)
$env:IAM_INPUT_ROOT = "dataset"
$env:VATR_WORK_DIR  = "work"
jupyter nbconvert --to notebook --execute dl-final-project.ipynb --output run_local.ipynb --ExecutePreprocessor.timeout=-1

# Linux / macOS
IAM_INPUT_ROOT=dataset VATR_WORK_DIR=work \
  jupyter nbconvert --to notebook --execute dl-final-project.ipynb --output run_local.ipynb --ExecutePreprocessor.timeout=-1
```

Environment variables read by the notebook:

| Variable | Default | Meaning |
|---|---|---|
| `IAM_INPUT_ROOT` | `/kaggle/input` | folder that contains the IAM files |
| `VATR_WORK_DIR` | `/kaggle/working` | output folder (checkpoints, cache, evaluation, samples) |
| `VATR_DEBUG` | `0` | `1` = short smoke test (a few minutes) to check that everything runs |
| `VATR_WORKERS` | `4` | data-loader worker processes (use `0` on Windows if loading fails) |
| `VATR_PRETRAINED` | `1` | `1` = start ResNet-50 from ImageNet weights |

Use a new, empty work folder for a fresh run. Pointing `VATR_WORK_DIR` at `results/`
would resume from the existing checkpoint.

Quick smoke test before a full run:
```bash
VATR_DEBUG=1 IAM_INPUT_ROOT=dataset VATR_WORK_DIR=work_debug jupyter nbconvert --to notebook --execute dl-final-project.ipynb --output debug.ipynb
```

After training, copy these files from the work folder into `results/`:
`vatr_resnet50_generator.pt`, `recognizer.pt`, `cache/unifont.hex.gz`, `evaluation/`, `samples/`.

---

## 5. Testing and evaluation

Evaluation (notebook section 10) runs automatically at the end of training on the 207
unseen test writers and the 424 seen writers. It computes CER, WER, word accuracy,
character precision/recall/F1 and confusion matrix, FID, KID, Geometry Score,
writer-identification accuracy/F1/kappa/MCC/top-5, and unseen-writer retrieval.

Results are saved in `results/evaluation/`:
- `metrics.csv`, `metrics.json` - all metrics
- `per_character_metrics_unseen.csv`, `char_confusion_unseen.png`
- `writer_confusion_seen_*.png`, `writer_retrieval_unseen_generated.png`
- `training_losses.png`, `validation_curves.png`, `real_vs_generated.png`

To re-run only the evaluation for an existing model, open the notebook in Jupyter with
`VATR_WORK_DIR` pointing to a folder that contains `checkpoints/best_G.pt` and
`checkpoints/last.pt`, run sections 0-7 and the section 8 cells before the training loop, skip the
training-loop cell, then run sections 10-12. The evaluation loads `best_G.pt` (or the final weights if it
is missing).

Quick functional test of the trained model (no dataset needed):
```bash
cd streamlit_app
python -c "from model import HandwritingModel; m = HandwritingModel(); print('loaded on', m.device)"
```

---

## 6. Running the web application

```bash
cd streamlit_app
streamlit run app.py
```

Open http://localhost:8501 in the browser.

1. **Sidebar 1 - Handwriting style:** choose an unseen IAM writer, upload a photo of a
   handwritten page (about 20+ words, dark pen on white paper), or upload single-word images.
2. **Sidebar 2/3 - Settings:** seed, stroke variation, line width; preset
   *Match the handwriting (recommended)* adapts letter width, stroke thickness, slant and
   ink tone to the style samples.
3. Type the text and press **Generate handwriting**; download the result as PNG.

The app loads the model weights from `results/` (override with the environment variable
`VATR_RESULTS_DIR`). Two generators can be selected: *Averaged* (mean of the step-24k and
final weights, default) and *Best checkpoint* (step 24k, the model evaluated in the report).

---

## 7. Folder structure

```
DL_PROJECT/
|-- README.md                       this file
|-- requirements.txt                all Python dependencies
|-- Project_Summary.txt             short project summary (models, metrics, tech stack)
|-- dl-final-project.ipynb          training + evaluation + generation notebook (run 3, with outputs)
|-- dataset/                        IAM dataset (not in git, see section 3)
|   |-- ascii/  words/  lines/  sentences/  xml/
|-- results/                        all trained weights and outputs
|   |-- vatr_resnet50_generator.pt      best generator (step 24,000)
|   |-- vatr_resnet50_generator_avg.pt  averaged generator (used by default in the app)
|   |-- recognizer.pt                   CRNN recognizer (readability check in the app)
|   |-- my_handwriting.png              example output
|   |-- cache/                          unifont.hex.gz (glyphs), iam_h32.pkl (image cache)
|   |-- checkpoints/                    best_G.pt, last.pt (full training state)
|   |-- evaluation/                     metrics.csv, metrics.json, plots
|   |-- samples/                        sample sheets saved during training
|-- streamlit_app/
    |-- app.py                      MAIN FILE - Streamlit web interface
    |-- model.py                    model classes, loading, generation, post-processing
    |-- requirements.txt            dependencies for the app only
    |-- README.txt                  short app notes
    |-- samples/writer_XXX/         word images + texts.json of 20 unseen IAM writers
```

---

## 8. Results (run 3, best checkpoint)

| Metric | Unseen writers | Seen writers | Real images (reference) |
|---|---|---|---|
| CER (lower is better) | 0.0055 | 0.0034 | 0.0790 (unseen) |
| WER | 0.0218 | 0.0131 | 0.2214 (unseen) |
| FID | 51.08 | 53.34 | 4.73 (real vs real) |
| KID | 0.0246 | 0.0246 | - |
| Geometry Score | 0.0094 | 0.0045 | - |
| Writer-ID accuracy (424 writers) | - | 50.9 % | 72.2 % |
| Unseen-writer retrieval (204 writers) | 71.1 % | - | 100 % |
| SSIM (generated vs real, same word) | 0.141 | 0.142 | 0.187 (two real writers) |

CER and writer-ID are measured with networks that also guided training, so they are
optimistic; FID is not directly comparable with the VATr paper (32 px vs 64 px images,
much shorter training).

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not load the model` in the app | check that `results/vatr_resnet50_generator_avg.pt` or `vatr_resnet50_generator.pt` exists |
| Errors after updating `model.py` | stop Streamlit (Ctrl+C) and start it again |
| `CUDA out of memory` during training | reduce the batch size in the configuration cell |
| Data loader hangs on Windows | set `VATR_WORKERS=0` |
| Photo upload finds few words | use a flat, well-lit photo with dark ink and at least 20 words |

## 10. Reference

V. Pippi, S. Cascianelli, R. Cucchiara. *Handwritten Text Generation from Visual
Archetypes.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.
#   A I - b a s e d - p e r s o n a l i z e d - h a n d w r i t i n g - g e n e r a t o r  
 