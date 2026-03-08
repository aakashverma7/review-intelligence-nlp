# Review Intelligence with NLP

This project combines two coursework streams into one portfolio-ready NLP repository:

- fake review classification
- review helpfulness classification and regression

The cleaned version focuses on reusable preprocessing, reproducible train/test splits, model comparison, and metrics that are easy to explain in interviews.

Data not included in this repository.

## What This Repository Covers

- text cleaning with stopword removal and stemming
- TF-IDF feature extraction
- fake review detection with multiple classifiers
- helpfulness scoring based on Amazon review votes
- binary helpfulness classification
- continuous helpfulness regression

## Project Structure

- `src/review_analytics.py`: end-to-end training and evaluation script
- `data/README.md`: expected dataset files and schema
- `requirements.txt`: Python dependencies

## Datasets

This repository expects:

- a local fake reviews CSV
- either a local export of the Amazon reviews dataset or internet access for the Hugging Face `amazon_us_reviews` dataset

The original datasets are not committed here.

## Quick Start

```bash
pip install -r requirements.txt
python src/review_analytics.py --fake-reviews-path data/fake_reviews.csv --sample-size 50000
```

If you have a local helpfulness dataset export:

```bash
python src/review_analytics.py --fake-reviews-path data/fake_reviews.csv --amazon-reviews-path data/amazon_mobile_electronics.csv
```

## Outputs

The script writes results under `outputs/`:

- `fake_review_results.csv`
- `helpfulness_classification_results.csv`
- `helpfulness_regression_results.csv`

## Why This Repo Is Worth Showcasing

- It aligns well with a data scientist profile: NLP preprocessing, text representation, supervised learning, and evaluation.
- It demonstrates how to turn noisy notebook work into a cleaner applied ML project.
