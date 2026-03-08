# Data Notes

Data not included in this repository.

## Required Fake Review Dataset

Place a CSV such as `fake_reviews.csv` in this folder.

Expected columns:

- `text_` or `text`
- `label`

The original notebook used labels similar to `CG` and `OR`.

## Optional Local Helpfulness Dataset

You can place a CSV such as `amazon_mobile_electronics.csv` in this folder instead of downloading from Hugging Face.

Expected columns:

- `review_body`
- `helpful_votes`
- `total_votes`

If you do not provide this file, the script will try to download:

- `amazon_us_reviews`
- subset `Mobile_Electronics_v1_00`
