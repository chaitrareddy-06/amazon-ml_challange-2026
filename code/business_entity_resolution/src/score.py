import pandas as pd

def parse_ids(s):
    if pd.isna(s) or s == "":
        return set()
    return set(s.split(","))

def f_half(pred, truth):
    if len(truth) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    if len(pred) == 0:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(truth)
    if precision == 0 and recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def main():
    pred_df = pd.read_csv("output/matching_results_baseline.tsv", sep="\t", dtype=str).fillna("")
    truth_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

    pred_map = dict(zip(pred_df["source1_entity_id"], pred_df["matched_entity_ids"]))
    truth_map = dict(zip(truth_df["source1_entity_id"], truth_df["matched_entity_ids"]))

    scores = []
    for s1_id, truth_str in truth_map.items():
        pred_str = pred_map.get(s1_id, "")
        pred = parse_ids(pred_str)
        truth = parse_ids(truth_str)
        scores.append(f_half(pred, truth))

    avg = sum(scores) / len(scores)
    print(f"Entities scored: {len(scores)}")
    print(f"Mean F_0.5: {avg:.4f}")

if __name__ == "__main__":
    main()