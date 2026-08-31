"""Watch the live score distribution against the one the model was trained on.

A fitted model degrades quietly. Nothing throws, nothing 500s -- the scores simply
stop describing the world, and the first evidence is usually a business metric
weeks later. The incident service's real-data track shows the shape of it: a
detector configured to alert on 3% of points fires on 21%, 6% and 52% across three
real machines, purely because the test period is not the training period.

This is the cheap version of catching that: compare the distribution of scores
being produced now against a reference captured at training time.

**Population Stability Index.** For bins with expected proportion e and observed
proportion a:

    PSI = sum (a - e) * ln(a / e)

Conventional readings, which this uses and names rather than hiding in a constant:

    < 0.10   stable
    < 0.25   moderate shift, worth looking at
    >= 0.25  significant shift, the model is seeing a different population

PSI is chosen over a KS test because it is bin-based and therefore comparable
against a reference stored as bin proportions, which is all a training run needs
to leave behind. It also degrades gracefully: no p-value to misread, no assumption
that the reference is a known continuous distribution.

The window is bounded and in-memory. This monitors a process, not a fleet; a
restart loses the window, and that is an acceptable cost for having no
dependencies. What it is not is a substitute for retraining -- it says the input
moved, not what to do about it.
"""
import json
import math
import threading
from collections import deque
from pathlib import Path

# Below this many observations the PSI of a small sample is noise, so a verdict is
# refused rather than guessed.
MIN_SAMPLES = 50
DEFAULT_WINDOW = 500

STABLE = "stable"
MODERATE = "moderate_shift"
SIGNIFICANT = "significant_shift"
INSUFFICIENT = "insufficient_data"

MODERATE_THRESHOLD = 0.10
SIGNIFICANT_THRESHOLD = 0.25

# Guards ln(0) when a bin the reference expected never appears live, and vice
# versa. Small enough not to move a real PSI, large enough to keep it finite.
EPSILON = 1e-6


def build_categorical_reference(labels):
    """Reference distribution over predicted CLASSES, for a classifier.

    Numeric PSI over a classifier's confidence was tried first and abandoned: on
    a template-generated corpus confidence is bimodal -- near 1.0 on phrasings the
    model has effectively memorised, much lower on anything else -- so the metric
    swung wildly on ordinary traffic and reported drift that was not there. A
    monitor that cries wolf trains people to ignore it.

    The mix of predicted classes is the robust signal, and the standard one for
    classifiers: if the proportion of refund requests doubles, something changed,
    and it says so without depending on how confident the model happens to feel.
    """
    counts = {}
    for label in labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    total = float(sum(counts.values()))
    if not total:
        raise ValueError("no labels to build a drift reference from")
    classes = sorted(counts)
    return {
        "kind": "categorical",
        "classes": classes,
        "proportions": [counts[name] / total for name in classes],
        "n_reference": int(total),
    }


def build_reference(scores, bins=10):
    """Reference distribution from training scores, for a training run to save.

    Quantile bins rather than equal-width: propensity scores cluster, and equal
    width would put almost everything in one bin and make PSI insensitive to
    exactly the movement it exists to detect.
    """
    ordered = sorted(float(score) for score in scores)
    if not ordered:
        raise ValueError("no scores to build a drift reference from")

    edges = [ordered[0]]
    for index in range(1, bins):
        position = int(len(ordered) * index / bins)
        edges.append(ordered[min(position, len(ordered) - 1)])
    edges.append(ordered[-1])
    # Deduplicate: heavily tied scores can produce repeated edges, which would
    # create empty bins that only add noise.
    edges = sorted(set(edges))

    counts = [0] * (len(edges) - 1)
    for score in ordered:
        counts[_bin_index(score, edges)] += 1
    total = float(len(ordered))
    return {
        "edges": edges,
        "proportions": [count / total for count in counts],
        "n_reference": len(ordered),
    }


def _bin_index(score, edges):
    """Index of the bin a score falls in; the tails are clamped inward.

    Live scores outside the training range are real and must be counted, not
    dropped -- a model seeing values it never saw in training is precisely the
    case this is for.
    """
    for index in range(len(edges) - 2, -1, -1):
        if score >= edges[index]:
            return index
    return 0


def population_stability_index(reference, observed_counts):
    total = sum(observed_counts)
    if total == 0:
        return 0.0
    psi = 0.0
    for expected, count in zip(reference["proportions"], observed_counts):
        actual = count / total
        expected = max(expected, EPSILON)
        actual = max(actual, EPSILON)
        psi += (actual - expected) * math.log(actual / expected)
    return psi


class DriftMonitor:
    """Rolling window of live scores, compared against a training reference."""

    def __init__(self, reference=None, window=DEFAULT_WINDOW, name="score"):
        self.reference = reference
        self.name = name
        self.window = window
        self._scores = deque(maxlen=window)
        self._lock = threading.Lock()
        self._total_seen = 0

    @classmethod
    def from_file(cls, path, **kwargs):
        """Load a reference written by a training run; None if absent.

        A missing reference is not an error: a service should still serve when
        drift monitoring is unavailable, and say so rather than refusing traffic.
        """
        path = Path(path)
        if not path.exists():
            return cls(reference=None, **kwargs)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(reference=data, **kwargs)

    def observe(self, score):
        """Record one observation: a number for a score, a string for a class."""
        with self._lock:
            self._scores.append(
                score if isinstance(score, str) else float(score)
            )
            self._total_seen += 1

    def report(self):
        with self._lock:
            scores = list(self._scores)
            total_seen = self._total_seen

        if self.reference is None:
            return {
                "status": "no_reference",
                "detail": ("no drift reference was saved at training time, so live "
                           "scores cannot be compared to anything"),
                "observed": len(scores),
                "total_seen": total_seen,
            }
        if len(scores) < MIN_SAMPLES:
            return {
                "status": INSUFFICIENT,
                "detail": f"need {MIN_SAMPLES} observations, have {len(scores)}",
                "observed": len(scores),
                "total_seen": total_seen,
                "min_samples": MIN_SAMPLES,
            }

        if self.reference.get("kind") == "categorical":
            classes = self.reference["classes"]
            index = {name: position for position, name in enumerate(classes)}
            counts = [0] * len(classes)
            unseen = 0
            for score in scores:
                position = index.get(str(score))
                if position is None:
                    # A class the reference never saw is itself drift, but it
                    # cannot be scored against a bin that does not exist. Counted
                    # and reported rather than silently dropped.
                    unseen += 1
                else:
                    counts[position] += 1
        else:
            edges = self.reference["edges"]
            counts = [0] * (len(edges) - 1)
            unseen = 0
            for score in scores:
                counts[_bin_index(score, edges)] += 1

        psi = population_stability_index(self.reference, counts)
        if psi >= SIGNIFICANT_THRESHOLD:
            status = SIGNIFICANT
        elif psi >= MODERATE_THRESHOLD:
            status = MODERATE
        else:
            status = STABLE

        return {
            "status": status,
            "metric": "population_stability_index",
            "psi": round(psi, 4),
            "thresholds": {
                "moderate": MODERATE_THRESHOLD,
                "significant": SIGNIFICANT_THRESHOLD,
            },
            "observed": len(scores),
            "total_seen": total_seen,
            "window": self.window,
            "n_reference": self.reference.get("n_reference"),
            "reference_bins": len(counts),
            "unseen_classes": unseen,
            "note": ("a shift says the input distribution moved, not that the model "
                     "is wrong and not what to do about it; retraining is a human "
                     "decision this does not make"),
        }
