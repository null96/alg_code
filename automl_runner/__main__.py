from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


DEFAULT_TRAIN_FILE = "anzhen2022_2024_move_cut_features_train.csv"
DEFAULT_TEST_FILE = "anzhen2022_2024_move_cut_features_test.csv"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal AF classification runner for OpenClaw local execution.")
    parser.add_argument("--data-dir", required=True, help="Local directory containing the CSV files.")
    parser.add_argument("--phase", default="smoke", choices=["inspect", "smoke"], help="Validation phase to run.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE, help="Training CSV filename relative to data-dir.")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="Test CSV filename relative to data-dir.")
    parser.add_argument("--target", default="af_annotation_label", help="Target column name.")
    parser.add_argument(
        "--drop-cols",
        default="file_name,source_record,dataset_split",
        help="Comma-separated columns to exclude from features if present.",
    )
    parser.add_argument("--output-json", default="", help="Optional path to write a JSON summary.")
    return parser


def _resolve_csv(data_dir: Path, filename: str) -> Path:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"missing CSV file: {path}")
    return path


def _load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _feature_columns(df: pd.DataFrame, target: str, drop_cols: list[str]) -> list[str]:
    excluded = {target, *drop_cols}
    numeric_cols = [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]
    return numeric_cols


def _inspect(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str, drop_cols: list[str]) -> dict[str, object]:
    features = _feature_columns(train_df, target, drop_cols)
    payload: dict[str, object] = {
        "ok": True,
        "runner": "automl_runner",
        "phase": "inspect",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_cols": int(train_df.shape[1]),
        "test_cols": int(test_df.shape[1]),
        "target": target,
        "feature_count": len(features),
        "feature_preview": features[:15],
        "missing_target_train": int(train_df[target].isna().sum()) if target in train_df.columns else None,
        "missing_target_test": int(test_df[target].isna().sum()) if target in test_df.columns else None,
    }
    if target in train_df.columns:
        payload["train_class_counts"] = {str(key): int(value) for key, value in train_df[target].value_counts().to_dict().items()}
    if target in test_df.columns:
        payload["test_class_counts"] = {str(key): int(value) for key, value in test_df[target].value_counts().to_dict().items()}
    return payload


def _smoke(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str, drop_cols: list[str]) -> dict[str, object]:
    features = _feature_columns(train_df, target, drop_cols)
    if not features:
        raise ValueError("no numeric feature columns available after exclusions")
    if target not in train_df.columns or target not in test_df.columns:
        raise ValueError(f"target column not found in both train and test data: {target}")

    x_train = train_df[features]
    y_train = train_df[target]
    x_test = test_df[features]
    y_test = test_df[target]

    preprocessor = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), features)],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)

    labels = np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_test)]))
    report = classification_report(y_test, preds, labels=labels, output_dict=True, zero_division=0)
    payload: dict[str, object] = {
        "ok": True,
        "runner": "automl_runner",
        "phase": "smoke",
        "target": target,
        "feature_count": len(features),
        "feature_preview": features[:15],
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "metrics": {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
        },
        "classification_report": report,
    }
    return payload


def _write_output(payload: dict[str, object], output_json: str) -> None:
    if not output_json:
        return
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = _resolve_csv(data_dir, args.train_file)
    test_path = _resolve_csv(data_dir, args.test_file)
    train_df = _load_frame(train_path)
    test_df = _load_frame(test_path)
    drop_cols = [item.strip() for item in args.drop_cols.split(",") if item.strip()]

    if args.phase == "inspect":
        payload = _inspect(train_df, test_df, args.target, drop_cols)
    else:
        payload = _smoke(train_df, test_df, args.target, drop_cols)

    _write_output(payload, args.output_json)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
