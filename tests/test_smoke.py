"""Behavioural tests for the diabetes progression predictor.

train_model.py trains at import time, so it is never imported here: its
constants are checked via ast and the CSV is exercised through load_data.
"""
import ast
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSV = ROOT / "data" / "diabetes.csv"
TARGET = "disease_progression"


def _module_assignments(path, names):
    """Return {name: value} for module-level literal/simple assignments."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in names:
                try:
                    out[tgt.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    out[tgt.id] = node.value
    return out


def test_readme_and_license_exist():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "LICENSE").is_file()


def test_target_constant_in_source():
    values = _module_assignments(ROOT / "src" / "train_model.py", {"TARGET"})
    assert values["TARGET"] == TARGET


def test_features_are_computed_by_dropping_target():
    """FEATURES is a comprehension over columns != TARGET, not a hardcoded list."""
    tree = ast.parse((ROOT / "src" / "train_model.py").read_text(encoding="utf-8"))
    features = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "FEATURES" for t in node.targets
        ):
            features = node.value
    assert isinstance(features, ast.ListComp)
    assert ast.unparse(features.elt) == "c"
    comp = features.generators[0]
    assert isinstance(comp.iter, ast.Attribute)
    assert ast.unparse(comp.iter) == "df.columns"
    assert ast.unparse(comp.ifs[0]) == "c != TARGET"


def test_saved_plot_filename_is_importance_png():
    tree = ast.parse((ROOT / "src" / "train_model.py").read_text(encoding="utf-8"))
    saved = [
        ast.literal_eval(call.args[0])
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "savefig"
        and call.args
    ]
    assert "importance.png" in saved


def test_csv_shape_and_no_missing_values():
    if not CSV.is_file():
        pytest.skip("cached data/diabetes.csv is not present")
    df = pd.read_csv(CSV)
    assert df.shape == (442, 11)
    assert TARGET in df.columns
    assert df.isna().sum().sum() == 0


def test_exactly_ten_feature_columns_remain():
    if not CSV.is_file():
        pytest.skip("cached data/diabetes.csv is not present")
    df = pd.read_csv(CSV)
    features = [c for c in df.columns if c != TARGET]
    assert len(features) == 10
    assert TARGET not in features


def test_load_returns_frame_with_target(monkeypatch):
    if not CSV.is_file():
        pytest.skip("cached data/diabetes.csv is not present")
    monkeypatch.chdir(ROOT)  # CSV_PATH is relative to the project root
    from src.load_data import CSV_PATH, load

    assert pathlib.Path(CSV_PATH).parent.name == "data"
    df = load()
    assert len(df) == 442
    assert TARGET in df.columns


def test_tiny_forest_predicts_finite_floats():
    if not CSV.is_file():
        pytest.skip("cached data/diabetes.csv is not present")
    from sklearn.ensemble import RandomForestRegressor

    df = pd.read_csv(CSV)
    features = [c for c in df.columns if c != TARGET]
    model = RandomForestRegressor(n_estimators=5, random_state=0)
    model.fit(df[features].iloc[:100], df[TARGET].iloc[:100])
    preds = model.predict(df[features].iloc[:100].head(7))
    assert preds.shape == (7,)
    assert np.isfinite(preds).all()