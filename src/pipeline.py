"""End-to-end run: clean -> time split -> CV on train -> single test evaluation.

Usage:
    python -m src.pipeline --data-path data/creditcard.csv --output-dir outputs

Model selection uses cross-validation inside the training part only. The test set is
used once, at the end, for the final model.
"""
import argparse
from pathlib import Path

import pandas as pd

from src import data, evaluate, explain, models


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Credit card fraud detection pipeline")
    p.add_argument("--data-path", required=True, help="Path to creditcard.csv")
    p.add_argument("--output-dir", default="outputs", help="Where to write tables and figures")
    p.add_argument("--test-frac", type=float, default=0.2, help="Share of the latest rows used as test")
    p.add_argument("--n-boot", type=int, default=500, help="Bootstrap resamples for the test CI")
    p.add_argument("--skip-cv", action="store_true", help="Skip the cross-validation stage")
    p.add_argument("--skip-shap", action="store_true", help="Skip the SHAP stage")
    return p.parse_args(argv)


def run_cross_validation(X_train, y_train):
    """Both CV schemes for baseline and SMOTE/Isolation Forest models (train only)."""
    results = []
    for builder in (models.build_models, models.build_models_v2):
        results.append(
            evaluate.run_cv(builder, evaluate.stratified_splitter(), X_train, y_train,
                            evaluate.SCHEME_STRATIFIED)
        )
        results.append(
            evaluate.run_cv(builder, evaluate.time_splitter(), X_train, y_train,
                            evaluate.SCHEME_TIME)
        )
    return pd.concat(results, ignore_index=True)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = data.load_data(args.data_path)
    df_clean = data.clean_data(df)
    train_df, test_df = data.time_split(df_clean, args.test_frac)
    X_train, y_train = data.split_features_target(train_df)
    X_test, y_test = data.split_features_target(test_df)
    print(f"Rows: {len(df)} -> {len(df_clean)} after removing {len(df) - len(df_clean)} duplicates")
    print(f"Train: {len(train_df)} rows, {int(y_train.sum())} frauds | "
          f"Test: {len(test_df)} rows, {int(y_test.sum())} frauds")

    if not args.skip_cv:
        cv = run_cross_validation(X_train, y_train)
        cv.to_csv(out / "cv_folds.csv", index=False)
        summary = evaluate.summarize_cv(cv)
        summary.to_csv(out / "cv_summary.csv")
        print("\nCross-validation (train only):")
        print(summary.to_string())

    # Final model: LightGBM without SMOTE, fitted on the training part only.
    final = models.make_lgbm().fit(X_train, y_train)
    p_test = final.predict_proba(X_test)[:, 1]
    y_arr = y_test.to_numpy()
    amount_test = X_test["Amount"].to_numpy()

    ci = evaluate.bootstrap_ci(y_arr, p_test, n_boot=args.n_boot)
    ci.round(3).to_csv(out / "test_metrics_bootstrap.csv")
    print("\nFinal test metrics (95% bootstrap CI, independent resampling of transactions):")
    print(ci.round(3).to_string())

    capacity = evaluate.capacity_table(y_arr, p_test, amount_test)
    capacity.to_csv(out / "capacity_table.csv", index=False)
    print("\nAnalyst capacity:")
    print(capacity.to_string(index=False))

    print("\nBest review share under assumed per-review costs (units of Amount):")
    for cost in (1, 5, 25):
        q, value = evaluate.best_review_fraction(y_arr, p_test, amount_test, cost)
        print(f"  cost={cost}: best share {q * 100:.2f}% -> net benefit {value:.0f}")
    evaluate.plot_net_benefit(y_arr, p_test, amount_test, path=out / "net_benefit.png")

    if not args.skip_shap:
        vals = explain.shap_values(final, X_test)
        importance = explain.mean_abs_shap(vals, X_test.columns, y_arr)
        importance.round(3).to_csv(out / "shap_mean_abs.csv")
        print("\nTop 10 features by mean |SHAP|, frauds only:")
        print(explain.top_features(importance, "Frauds only").to_string())
        explain.plot_summary(vals, X_test, path=out / "shap_summary.png")


if __name__ == "__main__":
    main()
