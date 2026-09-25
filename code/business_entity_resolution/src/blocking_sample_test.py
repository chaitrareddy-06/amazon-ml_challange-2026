import pandas as pd
import time
from blocking import build_candidates

t0 = time.time()
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t").head(20000)
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
print("loaded", time.time()-t0, flush=True)

result = build_candidates(s1, s2, s3, max_postings=300, max_candidates=40, chunk_size=5000)
print("built", time.time()-t0, flush=True)

result.to_csv("output/candidate_pairs_sample.tsv", sep="\t", index=False)
print("Done. Rows:", len(result), time.time()-t0)

avg_cands = result['candidate_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0).mean()
print("Avg candidates per S1 entity:", avg_cands)