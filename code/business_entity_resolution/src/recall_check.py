import pandas as pd

def parse_ids(s):
    if pd.isna(s) or s == "":
        return set()
    return set(s.split(","))

cand_df = pd.read_csv("output/candidate_pairs.tsv", sep="\t", dtype=str).fillna("")
truth_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

cand_map = dict(zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"]))
truth_map = dict(zip(truth_df["source1_entity_id"], truth_df["matched_entity_ids"]))

total_true = 0
found_true = 0

for s1_id, truth_str in truth_map.items():
    truth = parse_ids(truth_str)
    if not truth:
        continue
    cand = parse_ids(cand_map.get(s1_id, ""))
    total_true += len(truth)
    found_true += len(truth & cand)

print(f"Recall of blocking: {found_true/total_true:.4f}")