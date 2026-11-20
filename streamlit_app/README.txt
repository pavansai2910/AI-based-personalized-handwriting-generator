PERSONALIZED HANDWRITING GENERATOR - STREAMLIT WEB INTERFACE
=============================================================

Folder layout (already set up inside DL_PROJECT):
  DL_PROJECT/
    results/                               <- ALL model files (.pt) live here
      vatr_resnet50_generator_avg.pt       averaged generator (default in the app)
      vatr_resnet50_generator.pt           best generator of run 3 (step 24,000)
      recognizer.pt                        handwriting recognizer from training (33 MB):
                                           "versions tried per word" + readability check
      cache/unifont.hex.gz            Report check (I didn’t edit the report)

These claims are false or inaccurate:

Abstract: it calls VATr a “Vision-Transformer-style architecture”. VATr stands for Visual Archetypes-based Transformer; it is not a ViT.
Reference [1]: the authors are written as “F. Pippi, M. Cascianelli”. The correct names are V. Pippi, S. Cascianelli, R. Cucchiara.
Sec 4.7: W is called “the independent writer classifier”. Sections 6.3/6.7 say W supplies the generator’s writer loss, so it is not independent.
Sec 6.4: “all six rows spell the sentence correctly” (Fig. 9) is wrong. Rows 1 and 6 read “Th quick … over th lazy dog”, with the final “e” missing.
Sec 6.4: “Strokes … are thin, grey and blurred” overgeneralises. In Fig. 13 the generated “changes” is thick and dark.
Sec 4.10: retrieval did not use “15 generated words” for “every unseen writer”. It used 3–15 words per writer, and covered 204 of the 207 writers (only those with at least 33 words).
Sec 6.2 (minor): it says the CTC loss was “about 0.02–0.07 throughout”. About 77% of the logged values fall in that range, but the real range was 0.019–0.53.

All the dataset and split numbers, Tables 4–7, the run 1 and run 2 numbers, the hyperparameters and the step-24k figure are correct.

The report also leaves out SSIM, the Streamlit app and the averaged generator.     character glyphs
      checkpoints/best_G.pt, last.pt       training checkpoints (only for retraining)
    streamlit_app/
      app.py             the web page (Streamlit)
      model.py           model architecture + generation code (inference only)
      requirements.txt   Python packages
      samples/           15 word images for each of 20 IAM writers that were
                         NOT used in training (unseen styles) + texts.json (what
                         each word says, used to measure the exact letter width)

The app loads every model file only from DL_PROJECT/results/ (the folder next to
streamlit_app). To use another location, set the environment variable
VATR_RESULTS_DIR to that folder before starting the app.

HOW TO RUN (Windows / macOS / Linux)
  1. Install Python 3.9 - 3.12.
  2. Open a terminal in the streamlit_app folder:
        cd "C:\Users\MOGILI JYOTHI\Downloads\DL_PROJECT\streamlit_app"
  3. (optional) create a virtual environment:
        python -m venv venv
        venv\Scripts\activate            (Windows)
  4. Install the packages:
        pip install -r requirements.txt
  5. Start the app:
        streamlit run app.py
     Your browser opens at http://localhost:8501
  A GPU is not needed: on a normal CPU a sentence takes about 1-3 seconds.

HOW TO USE
  Sidebar - 1. Handwriting style (choose one):
     * IAM writer        pick one of the 20 sample writers
     * Photo of a page   upload a photo/scan of handwriting (~20+ words, dark pen,
                         white paper); detected words are shown with red boxes
     * Word images       upload several images of single handwritten words
  Sidebar - Generator weights:
     * Averaged (recommended)  mean of the weights of the best generator (step 24,000)
                               and the final generator (step 45,879) of run 3 ("model
                               soup"): fewer stray marks, slightly stronger style
     * Best checkpoint         the model evaluated in the notebook (step 24,000)
  Sidebar - 2. Settings:
     * Random seed       different seeds give slightly different strokes
     * Stroke variation  amount of random noise (0.2 = as trained)
     * Line width        where the text wraps to a new line
     * Clean background  removes the faint grey haze around letters
     * Output zoom       size of the result image
  Sidebar - 3. Appearance, Preset:
     * Match the handwriting (recommended, default) - for the closest resemblance:
         - letter width : measured from the style samples (exactly for the IAM sample
                          writers via samples/<writer>/texts.json, estimated by the
                          recognizer for your own uploads) and applied to every word
         - stroke width : the thickness setting whose strokes are closest to the
                          writer's real pen strokes is chosen automatically
         - slant        : the writer's slant is measured and applied
         - ink tone     : the writer's own ink darkness
         - faint letters boosted, sharp rendering, clean full stops/commas
       The settings that were applied are shown under the result.
     * Most readable  - medium letter width (0.8), normal thickness, black ink
     * Custom         - set letter spacing, thickness, slant, ink colour etc. yourself
     * Versions tried per word: writes each word several times and keeps the clearest one
     * Show raw model output: for before/after comparison
  Main area: type the text, press "Generate handwriting", then "Download PNG".
  "Readability check" shows what the training recognizer reads in the result.

KNOWN LIMITATIONS
  * Output height is 32 px per line (scaled up for viewing), so strokes look soft.
  * Punctuation (especially '.') is rendered less cleanly than letters.
  * Characters outside the 79 training characters may look wrong.
