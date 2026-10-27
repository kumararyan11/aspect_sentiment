# Sentiment & Aspect-Based Analysis of Product Reviews (DistilBERT)

6 aspects: price, quality, delivery, packaging, customer_service, overall_experience.
Each aspect gets a 4-way label: not_mentioned / negative / neutral / positive.

## Pipeline
raw review -> custom cleaning (emoji, slang, Hinglish, elongation, regex) -> DistilBERT (shared encoder, 6 heads) -> per-aspect sentiment

## Run
    pip install -r requirements.txt
    # data: either real reviews (weak-labelled) ...
    python prepare_data.py weak --input raw_reviews.csv --text_col review --rating_col rating
    # ... or synthetic data to test the pipeline
    python prepare_data.py synthetic --n 8000

    python baseline.py                 # TF-IDF + LogReg
    python train.py --no_clean         # DistilBERT, raw text   (ablation)
    python train.py                    # DistilBERT, cleaned text
    python compare.py                  # benchmark table + F1 delta from preprocessing
    streamlit run app.py

## Notes
- Report numbers from compare.py on YOUR data. Weak labels cap accuracy; hand-correct a test subset if possible.
- For heavily multilingual data: python train.py --model_name distilbert-base-multilingual-cased
