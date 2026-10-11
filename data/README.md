# Data

The dataset is **not** included in this repository.

- Source: Kaggle, [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group)
- File: `creditcard.csv` (284,807 transactions, 31 columns, target `Class`)

## Download

With the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) (requires an API token in `~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
```

This places `creditcard.csv` in this folder. `data/*.csv` is git-ignored.

## Use

```bash
python -m src.pipeline --data-path data/creditcard.csv
```

Please read the dataset's license and citation terms on its Kaggle page before reuse.
