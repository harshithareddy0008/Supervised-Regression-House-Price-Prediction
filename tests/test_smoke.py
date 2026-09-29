import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import RAW_DATA
from data_preprocessing import load_data, split_features_target, build_preprocessor


def test_dataset_and_preprocessor():
    df = load_data(RAW_DATA)
    assert len(df) >= 1000
    X, y = split_features_target(df)
    assert len(X) == len(y)
    pre = build_preprocessor(scale_numeric=True)
    Xt = pre.fit_transform(X.head(100))
    assert Xt.shape[0] == 100
    assert Xt.shape[1] > X.shape[1]
