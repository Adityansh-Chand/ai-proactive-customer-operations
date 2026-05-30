
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestration.agent_dag import run_dag


df = pd.read_csv(ROOT / "datasets" / "sample_data.csv")

correct = 0
for row in df.to_dict("records"):
    result = run_dag(row["message"], customer_id=row["customer_id"])
    correct += result["policy"] == row["expected_policy"] and result["action"]["type"] == row["expected_action"]

print("records:", len(df))
print("policy_action_accuracy:", correct / len(df))
