# Fake Review Detection and Helpfulness Prediction

This project applies classical NLP and machine learning techniques to two review-analysis tasks:

- fake review classification
- review helpfulness classification and regression

Data not included in this repository.

## Overview

- text preprocessing with stopword removal and stemming
- TF-IDF feature extraction
- fake review detection with multiple classifiers
- helpfulness prediction from Amazon review votes
- binary helpfulness classification
- continuous helpfulness regression

## Project Structure

- `src/review_analytics.py`: end-to-end training and evaluation script
- `data/README.md`: expected dataset files and schema
- `requirements.txt`: Python dependencies

## Data

This repository expects:

- a local fake reviews CSV
- either a local export of the Amazon reviews dataset or internet access for the Hugging Face `amazon_us_reviews` dataset

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
