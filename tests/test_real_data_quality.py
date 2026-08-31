"""Quality gate for the real-data intent track (BANKING77).

Skips cleanly when the dataset is not cached, so a fresh clone stays green
without a network call. CI fetches it, so CI runs these.

Observed at the time of writing: macro-F1 0.9164, accuracy 0.9162 on the official
77-class test split, with 6 verbatim train/test duplicates (0.195% of test).
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.fetch_real_data import CACHE_DIR  # noqa: E402

METRICS_PATH = ROOT / "models" / "artifacts" / "real" / "metrics.json"

MIN_MACRO_F1 = 0.86
MIN_ACCURACY = 0.86
# Verbatim overlap above this stops being incidental and starts explaining the
# score. The official files carry 0.195%; a jump means the upstream data changed.
MAX_OVERLAP_PCT = 1.0


pytestmark = pytest.mark.skipif(
    not (CACHE_DIR / "train.csv").exists() or not METRICS_PATH.exists(),
    reason="BANKING77 not cached; run training/fetch_real_data.py then train_real.py",
)


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def test_uses_the_official_split(metrics):
    """Re-splitting would make these numbers incomparable to published work."""
    assert "official" in metrics["split"]
    assert metrics["n_train"] == 10003
    assert metrics["n_test"] == 3080
    assert metrics["n_classes"] == 77


def test_macro_f1_holds(metrics):
    assert metrics["test_macro_f1"] >= MIN_MACRO_F1
    assert metrics["test_accuracy"] >= MIN_ACCURACY


def test_beats_trivial_baselines_by_a_wide_margin(metrics):
    """77 classes: the baselines are near zero, so this is a floor, not a win."""
    baselines = metrics["baselines"]
    assert metrics["test_accuracy"] > baselines["random_choice_accuracy"] * 10
    assert metrics["test_accuracy"] > baselines["majority_class_accuracy"] * 10


def test_leakage_stays_negligible(metrics):
    """The score is high; this is the check that says why it is trustworthy."""
    overlap = metrics["leakage_check"]["overlap_pct_of_test"]
    assert overlap <= MAX_OVERLAP_PCT, (
        f"verbatim train/test overlap rose to {overlap}% -- at this level the "
        "score is partly memorisation and the headline must be requalified"
    )


def test_hard_classes_are_reported(metrics):
    """A per-class breakdown is the point; an aggregate alone hides the failures."""
    assert len(metrics["hardest_classes"]) == 5
    assert all(row["f1"] < 1.0 for row in metrics["hardest_classes"])
    assert len(metrics["per_class_f1"]) == 77


def test_scope_is_stated_as_intent_only(metrics):
    """BANKING77 has no sentiment labels; the artifact must not imply otherwise."""
    assert "intent only" in metrics["component"]


def test_real_data_is_attributed(metrics):
    """CC BY 4.0 requires attribution, so the artifact carries it."""
    assert "REAL" in metrics["data_source"]
    assert metrics["citation"]
    assert "banking77" in metrics["url"]
