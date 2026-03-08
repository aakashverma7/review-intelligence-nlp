import argparse
import re
from pathlib import Path
from typing import Optional

import nltk
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


def normalize_text(text: str, stop_words: set[str], stemmer) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [token for token in text.split() if token and token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)


def load_fake_review_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = "text_" if "text_" in df.columns else "text"
    if text_col not in df.columns or "label" not in df.columns:
        raise ValueError("Fake review dataset must contain `text_` or `text`, plus `label`.")
    return df[[text_col, "label"]].rename(columns={text_col: "text"}).dropna()


def load_amazon_reviews(local_path: Optional[Path], sample_size: int) -> pd.DataFrame:
    if local_path:
        df = pd.read_csv(local_path)
    else:
        from datasets import load_dataset

        dataset = load_dataset("amazon_us_reviews", "Mobile_Electronics_v1_00", split="train")
        df = dataset.to_pandas()

    required = {"review_body", "helpful_votes", "total_votes"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Helpfulness dataset is missing columns: {sorted(missing)}")

    df = df[list(required)].dropna()
    df = df[df["total_votes"] > 0].copy()
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    return df


def evaluate_classifiers(models: dict[str, object], x_train, x_test, y_train, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision_weighted": precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "recall_weighted": recall_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
            }
        )
    return pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)


def evaluate_regressors(models: dict[str, object], x_train, x_test, y_train, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        rows.append(
            {
                "model": name,
                "rmse": rmse,
                "mae": mean_absolute_error(y_test, predictions),
                "r2": r2_score(y_test, predictions),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse")


def run_fake_review_classification(df: pd.DataFrame, output_dir: Path) -> None:
    stop_words = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()
    df = df.copy()
    df["clean_text"] = df["text"].map(lambda value: normalize_text(value, stop_words, stemmer))

    x_train, x_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2, max_df=0.8)
    x_train_matrix = vectorizer.fit_transform(x_train)
    x_test_matrix = vectorizer.transform(x_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "linear_svc": LinearSVC(),
        "multinomial_nb": MultinomialNB(),
    }

    results = evaluate_classifiers(models, x_train_matrix, x_test_matrix, y_train, y_test)
    results.to_csv(output_dir / "fake_review_results.csv", index=False)


def run_helpfulness_modeling(df: pd.DataFrame, output_dir: Path) -> None:
    stop_words = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()
    df = df.copy()
    df["clean_review"] = df["review_body"].map(lambda value: normalize_text(value, stop_words, stemmer))
    df["fraction_helpful"] = 100.0 * df["helpful_votes"] / df["total_votes"]
    df["helpful"] = (df["fraction_helpful"] > 50.0).astype(int)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, min_df=5, max_df=0.9)
    review_matrix = vectorizer.fit_transform(df["clean_review"])

    if review_matrix.shape[1] <= 2:
        n_components = 1
    else:
        n_components = min(300, review_matrix.shape[1] - 1)
    reducer = TruncatedSVD(n_components=n_components, random_state=42)
    dense_features = reducer.fit_transform(review_matrix)

    x_train_cls, x_test_cls, y_train_cls, y_test_cls = train_test_split(
        dense_features,
        df["helpful"],
        test_size=0.2,
        random_state=42,
        stratify=df["helpful"],
    )

    classification_models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "linear_svc": LinearSVC(),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
    }
    classification_results = evaluate_classifiers(
        classification_models, x_train_cls, x_test_cls, y_train_cls, y_test_cls
    )
    classification_results.to_csv(output_dir / "helpfulness_classification_results.csv", index=False)

    x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
        dense_features,
        df["fraction_helpful"],
        test_size=0.2,
        random_state=42,
    )
    regression_models = {
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
    }
    regression_results = evaluate_regressors(
        regression_models, x_train_reg, x_test_reg, y_train_reg, y_test_reg
    )
    regression_results.to_csv(output_dir / "helpfulness_regression_results.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NLP models for review intelligence tasks.")
    parser.add_argument("--fake-reviews-path", required=True, help="Path to the fake reviews CSV file.")
    parser.add_argument(
        "--amazon-reviews-path",
        help="Optional local CSV for helpfulness modeling. If omitted, the Hugging Face dataset is used.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Maximum number of helpfulness rows to keep for reproducibility.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for metrics CSV files.")
    return parser.parse_args()


def main() -> None:
    ensure_nltk_resource("corpora/stopwords", "stopwords")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fake_reviews = load_fake_review_data(Path(args.fake_reviews_path))
    run_fake_review_classification(fake_reviews, output_dir)

    amazon_reviews_path = Path(args.amazon_reviews_path) if args.amazon_reviews_path else None
    amazon_reviews = load_amazon_reviews(amazon_reviews_path, args.sample_size)
    run_helpfulness_modeling(amazon_reviews, output_dir)

    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
