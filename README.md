# Aspect-Based Sentiment Analysis of Product Reviews

A multi-aspect sentiment analysis pipeline that predicts sentiment
separately for six aspects of a product review. The project includes
custom text normalization, a multi-head DistilBERT model, a TF-IDF +
Logistic Regression baseline, evaluation utilities, and a Streamlit
interface for single-review and batch-CSV inference.

## Overview

Traditional sentiment analysis often assigns one sentiment label to an
entire review. A review, however, may praise product quality while
complaining about delivery or packaging. This project predicts sentiment
independently for each configured aspect.

### Aspects and labels

The model predicts the following six aspects:

-   `price`
-   `quality`
-   `delivery`
-   `packaging`
-   `customer_service`
-   `overall_experience`

Each aspect receives one of four labels:

-   `not_mentioned` --- the review does not clearly discuss the aspect
-   `negative`
-   `neutral`
-   `positive`

The model uses a shared DistilBERT encoder with a separate four-class
output for each aspect.

## Features

-   **Multi-aspect prediction:** predicts six aspect labels from a
    single review.
-   **Custom text normalization:** handles HTML entities/tags, URLs,
    user mentions, emoji and emoticon sentiment, repeated punctuation,
    elongated words, common slang, contractions, and selected
    romanized-Hindi (Hinglish) expressions.
-   **Transformer model:** fine-tunes DistilBERT with a shared encoder
    and six aspect outputs.
-   **Class-imbalance handling:** uses square-root inverse-frequency
    class weights during training.
-   **Classical baseline:** trains a TF-IDF + Logistic Regression
    classifier per aspect, with a custom tokenizer that marks a limited
    scope of negation.
-   **Preprocessing ablation:** compares DistilBERT trained on cleaned
    text with DistilBERT trained on raw text.
-   **Evaluation:** reports overall accuracy, macro-F1 across all
    labels, and metrics on mentioned aspects, plus per-aspect macro-F1.
-   **Data preparation:** supports synthetic data for pipeline testing
    and weak labels from a real reviews CSV with star ratings.
-   **Interactive app:** analyzes one review or a CSV containing a
    `text` column; batch predictions can be downloaded as a CSV.
-   **Reproducible splits:** uses the configured random seed and the
    same train/validation/test split strategy across models.

## Pipeline architecture

``` text
Product review text
        |
        v
Text cleaning / normalization
(emoji, slang, Hinglish, URLs, elongation, punctuation)
        |
        v
DistilBERT tokenizer
        |
        v
Shared DistilBERT encoder
        |
        v
Attention-mask mean pooling
        |
        v
Dropout + linear output layer
        |
        v
6 aspect outputs × 4 sentiment labels
        |
        v
Aspect-wise label + confidence
```

For benchmarking, a TF-IDF + Logistic Regression baseline is also
included. It uses the same data split strategy and a custom tokenizer
with limited negation marking.

## Technology stack

  Component             Technology
  --------------------- ----------------------------------------------
  Language              Python
  Transformer encoder   Hugging Face Transformers / DistilBERT
  Deep learning         PyTorch
  Text processing       Python regex, Unicode normalization, `emoji`
  Classical baseline    scikit-learn TF-IDF + Logistic Regression
  Data handling         pandas, NumPy
  Progress display      tqdm
  User interface        Streamlit
  Evaluation            Accuracy, macro-F1, per-aspect macro-F1

## Repository structure

``` text
aspect_sentiment/
├── app.py              # Streamlit UI for single and batch prediction
├── baseline.py         # TF-IDF + Logistic Regression baseline
├── compare.py          # Comparison table for saved experiment metrics
├── config.py           # Aspect names, labels, paths, and hyperparameters
├── dataset.py          # Data loading, split creation, and batch collation
├── metrics.py          # Overall and per-aspect evaluation metrics
├── model.py            # Shared DistilBERT encoder and multi-aspect output layer
├── predict.py          # Inference wrapper for saved model checkpoints
├── prepare_data.py     # Synthetic data generation and weak-label generation
├── preprocess.py       # Text cleaning and baseline tokenizer
├── train.py            # DistilBERT fine-tuning and evaluation
├── requirements.txt    # Python dependencies
├── data/               # Generated or user-provided review data
├── outputs/            # Saved model checkpoints, tokenizer, and metadata
└── results/            # Saved JSON metrics for each experiment
```

Generated model files and evaluation results are created when the
corresponding commands run; they may not be present in a fresh copy of
the repository.

## Getting started

### 1. Requirements

-   Python 3.10+ recommended.
-   `pip`
-   Internet access on the first run to download the selected Hugging
    Face model and tokenizer.
-   A CUDA-compatible GPU is recommended for faster fine-tuning, but the
    code can run on CPU with reduced speed.

### 2. Set up the environment

Run these commands from the `aspect_sentiment/` project directory:

``` bash
python -m venv .venv
```

Activate the environment:

``` bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

## Prepare a dataset

The project expects a CSV with these columns:

``` text
text,price,quality,delivery,packaging,customer_service,overall_experience
```

Each aspect column must contain one of `not_mentioned`, `negative`,
`neutral`, or `positive`.

### Option A: Generate synthetic data

Synthetic data is useful for testing the pipeline end to end. It is
generated from templates with controlled noise; it should **not** be
treated as a substitute for real-world evaluation.

``` bash
python prepare_data.py synthetic --n 8000
```

By default, the output is written to `data/reviews.csv`.

### Option B: Weak-label real reviews

Prepare a source CSV with a review-text column and a star-rating column.
Then run:

``` bash
python prepare_data.py weak \
  --input raw_reviews.csv \
  --text_col review \
  --rating_col rating
```

Change `review` and `rating` to match your input CSV headers. You can
also set the output path:

``` bash
python prepare_data.py weak \
  --input raw_reviews.csv \
  --text_col review \
  --rating_col rating \
  --out data/reviews.csv
```

This mode assigns aspect labels using keyword matching and a sentiment
lexicon, and derives `overall_experience` from the star rating. These
are **weak labels**, not human annotations; label noise can limit
achievable performance. If possible, manually correct a representative
subset and evaluate on that subset.

## Train and evaluate

### 1. Train the TF-IDF baseline

``` bash
python baseline.py
```

Use a custom dataset path with:

``` bash
python baseline.py --data data/reviews.csv
```

### 2. Fine-tune DistilBERT on cleaned text

``` bash
python train.py
```

### 3. Run the raw-text ablation

``` bash
python train.py --no_clean
```

The cleaned and raw runs are saved separately under `outputs/` as
`distilbert_clean/` and `distilbert_raw/`.

Useful training options include:

``` bash
python train.py --data data/reviews.csv --epochs 4 --batch_size 32 --lr 3e-5
```

To use the multilingual DistilBERT checkpoint configured in the project
documentation:

``` bash
python train.py --model_name distilbert-base-multilingual-cased
```

The selected model name is stored with the saved run metadata. The
multilingual model is an optional configuration, not evidence that the
pipeline has been validated across all languages.

### 4. Compare saved experiments

After the baseline and the desired DistilBERT runs have completed:

``` bash
python compare.py
```

This prints a table for the metrics JSON files present in `results/`
and, when both relevant runs exist, reports the macro-F1 difference
between cleaned and raw DistilBERT and the accuracy gap between
DistilBERT and the TF-IDF baseline.

## Evaluation metrics

The evaluation code reports:

-   **`acc_all`:** accuracy over all aspect labels, including
    `not_mentioned`.
-   **`f1_all`:** macro-F1 over the four labels, including
    `not_mentioned`.
-   **`acc_mentioned`:** accuracy only on examples whose true label is
    not `not_mentioned`.
-   **`f1_mentioned`:** macro-F1 over negative, neutral, and positive
    labels on those mentioned examples.
-   **`per_aspect_f1`:** macro-F1 for each of the six aspects, including
    `not_mentioned`.

Because `not_mentioned` may be common, `acc_all` alone can hide weak
sentiment classification. Review `f1_mentioned` and per-aspect scores as
well. When reporting results, state the dataset source, label-generation
method, split, and whether the evaluation set was manually annotated. Do
not claim performance figures until the experiments have been run and
the metrics files inspected.

## Run inference

### Command line

After training the default cleaned-text model:

``` bash
python predict.py Great quality but delivery was very late
```

The script prints each aspect, its predicted label, and the model
confidence. Ensure that the checkpoint exists at
`outputs/distilbert_clean/` first.

### Streamlit interface

``` bash
streamlit run app.py
```

The interface has two tabs:

1.  **Single review:** enter a review and view each aspect's predicted
    sentiment and confidence.
2.  **Batch (CSV):** upload a CSV containing a `text` column, review the
    predicted aspect columns, inspect a sentiment-count chart, and
    download the annotated CSV.

The app uses the saved `distilbert_clean` run by default. Train that run
before launching the app.

## Configuration

Edit `config.py` to adjust the main settings.

  ------------------------------------------------------------------------
  Setting                                    Default Purpose
  --------------------- ---------------------------- ---------------------
  `ASPECTS`                     6 configured aspects Aspect names
                                                     predicted by the
                                                     model

  `LABELS`                                  4 labels Label order used by
                                                     training and
                                                     inference

  `model_name`             `distilbert-base-uncased` Hugging Face encoder
                                                     checkpoint

  `max_len`                                    `128` Maximum token
                                                     sequence length

  `batch_size`                                  `32` Training batch size

  `epochs`                                       `4` Maximum training
                                                     epochs

  `lr`                                        `3e-5` Learning rate

  `weight_decay`                              `0.01` AdamW weight decay

  `warmup_ratio`                              `0.06` Fraction of training
                                                     steps used for
                                                     learning-rate warmup

  `dropout`                                    `0.2` Dropout before the
                                                     output layer

  `patience`                                     `2` Early-stopping
                                                     patience based on
                                                     validation macro-F1

  `seed`                                        `42` Random seed for split
                                                     and initialization
  ------------------------------------------------------------------------

The split is 80% training, 10% validation, and 10% test. The best
checkpoint is selected by validation `f1_all`, and final metrics are
computed on the held-out test split.

## Important limitations

-   **Weak-label noise:** keyword and lexicon rules can miss context,
    sarcasm, mixed sentiment, and aspect-specific meaning.
-   **Synthetic-data bias:** template-generated reviews are useful for
    pipeline checks but do not represent the full diversity of real
    customer language.
-   **Language coverage:** the default model is English-focused.
    Hinglish normalization covers only a selected vocabulary and does
    not make the system generally multilingual.
-   **Aspect coverage:** aspect names and keyword rules are predefined;
    new domains may require new aspects and labeled examples.
-   **Confidence calibration:** the softmax maximum is a model
    confidence score, not a calibrated probability of correctness.
-   **Preprocessing trade-offs:** normalization can remove useful nuance
    or change meaning; the included ablation is intended to measure its
    effect on the chosen dataset.

## Future improvements

-   Build a manually annotated, stratified test set with clear aspect
    and sentiment guidelines.
-   Report confidence intervals and per-class precision/recall alongside
    macro-F1.
-   Evaluate on product categories and review sources not used during
    training.
-   Calibrate confidence scores and test abstention for low-confidence
    predictions.
-   Expand multilingual handling with language-specific data and
    evaluation.
-   Add automated tests for preprocessing, label mapping, CSV
    validation, and model output shapes.
-   Track model size, inference latency, memory use, and batch
    throughput.

## Responsible use

Use predictions as decision-support signals rather than ground truth.
Review errors before using outputs for customer-service escalation,
product quality decisions, or public reporting. Avoid using unreviewed
predictions to make high-impact decisions about individuals.

## License

No license file was present in the supplied project archive. Add a
license only after deciding the terms under which the code may be reused
and distributed.
