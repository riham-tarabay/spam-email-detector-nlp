# 📧 Spam Email Detector

> Full NLP pipeline: text preprocessing → TF-IDF → multi-model comparison with evaluation metrics.

## Results

| Model | Accuracy | F1 (Spam) | ROC-AUC |
|---|---|---|---|
| Linear SVM | 98.9% | 0.974 | 0.997 |
| Complement NB | 98.3% | 0.961 | 0.995 |
| Logistic Regression | 98.1% | 0.957 | 0.996 |
| Random Forest | 97.6% | 0.948 | 0.993 |

*(Results on SMS Spam Collection dataset)*

## Quick Start
```bash
pip install -r requirements.txt
python detector.py
```

## Pipeline
```
Raw Text → Lowercase → Remove URLs/emails/phones
         → Remove punctuation → Remove stopwords
         → TF-IDF (bigrams, 10k features) → Model
```

## What I Learned
- Custom text preprocessing vs sklearn's built-in
- Why Complement NB outperforms Multinomial NB on imbalanced text
- TF-IDF with bigrams captures phrases like "FREE entry", "WINNER!!"
- Class imbalance handling with `class_weight="balanced"`

## Tech Stack
`Python` · `scikit-learn` · `NLTK` · `Pandas` · `Matplotlib`
