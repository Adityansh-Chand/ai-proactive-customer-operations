"""Two-level evaluation on the held-out template split.

Component level: how well does each classifier recover its label?
System level:    with the deterministic rules applied to PREDICTED labels, how
                 often is the final policy and action right?

Both matter and they differ. A single classification error changes the
downstream decision, so end-to-end accuracy is always at or below the component
scores. Reporting only the component numbers would flatter the system.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.action_agent import action_for_policy  # noqa: E402
from agents.classifiers import load_artifacts  # noqa: E402
from agents.policy_agent import decide_policy  # noqa: E402
from training.train import template_split  # noqa: E402


def main():
    intent_model, sentiment_model, metadata = load_artifacts()
    frame = pd.read_csv(ROOT / "datasets" / "messages.csv")
    _, test_idx, held_out = template_split(frame)
    test = frame.iloc[test_idx]
    messages = test["message"].to_numpy()

    predicted_intent = intent_model.predict(messages)
    predicted_sentiment = sentiment_model.predict(messages)
    predicted_policy = [
        decide_policy(intent=i, sentiment=s, priority=p)
        for i, s, p in zip(predicted_intent, predicted_sentiment, test["priority"])
    ]
    predicted_action = [action_for_policy(p) for p in predicted_policy]

    print("Model       :", metadata["model_type"])
    print("Architecture:", metadata["architecture"])
    print("Data        :", metadata["data_source"], "(SYNTHETIC)")
    print("Split       :", metadata["split"])
    print(f"Held-out templates: {len(held_out)}  |  test rows: {len(test)}")
    print()

    print("=== component level ===")
    print("-- intent --")
    print(classification_report(test["intent"], predicted_intent, zero_division=0))
    print("-- sentiment --")
    print(classification_report(test["sentiment"], predicted_sentiment, zero_division=0))

    print("=== system level (rules applied to PREDICTED labels) ===")
    policy_accuracy = accuracy_score(test["expected_policy"], predicted_policy)
    action_accuracy = accuracy_score(test["expected_action"], predicted_action)
    print(f"policy accuracy : {policy_accuracy:.4f}")
    print(f"action accuracy : {action_accuracy:.4f}")
    print()

    labels = sorted(test["expected_policy"].unique())
    matrix = confusion_matrix(test["expected_policy"], predicted_policy, labels=labels)
    print("policy confusion matrix (rows=expected, cols=predicted)")
    width = max(len(l) for l in labels) + 2
    print(" " * width + "".join(f"{l[:12]:>14}" for l in labels))
    for name, row in zip(labels, matrix):
        print(f"{name:<{width}}" + "".join(f"{v:>14}" for v in row))
    print()

    contrast = metadata.get("naive_row_split_contrast", {})
    if contrast:
        print(f"For contrast: the same pipeline under a naive ROW-level split scores")
        print(f"{contrast['intent_macro_f1']} intent macro-F1. That number is inflated -- train and")
        print("test share phrasings there, so it measures memorisation, not generalisation.")


if __name__ == "__main__":
    main()
