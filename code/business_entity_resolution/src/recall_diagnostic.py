import pandas as pd
import re
import time

STOPWORDS = {'inc', 'llc', 'ltd', 'limited', 'corp', 'corporation',
             'llp', 'pvt', 'private', 'co', 'company', 'the', 'and'}

def tokenize(name):
    if pd.isna(name):
        return set()
    s = str(name).lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    return {t for t in s.split() if t not in STOPWORDS and len(t) > 1}

t0 = time.time()
print("loading files...", flush=True)
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")
cand = pd.read_csv("output/candidate_pairs.tsv", sep="\t")
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
print(f"loaded in {time.time()-t0:.1f}s, gt rows={len(gt)}", flush=True)

name_lookup = pd.concat([s1[['entity_id', 'business_name']],
                          s2[['entity_id', 'business_name']],
                          s3[['entity_id', 'business_name']]]).set_index('entity_id')['business_name'].to_dict()
print(f"name lookup built {time.time()-t0:.1f}s", flush=True)

# Build candidate sets per s1_id (itertuples is much faster than iterrows)
cand_map = {}
for row in cand.itertuples(index=False):
    ids = str(row.candidate_entity_ids).split(',') if pd.notna(row.candidate_entity_ids) and row.candidate_entity_ids != '' else []
    cand_map[row.source1_entity_id] = set(ids)
print(f"candidate map built {time.time()-t0:.1f}s", flush=True)

zero_token_misses = []
had_token_misses = []

count = 0
for row in gt.itertuples(index=False):
    count += 1
    if count % 200000 == 0:
        print(f"  processed {count}/{len(gt)} rows, {time.time()-t0:.1f}s elapsed", flush=True)

    s1_id = row.source1_entity_id
    if pd.isna(row.matched_entity_ids) or row.matched_entity_ids == '':
        continue
    true_ids = str(row.matched_entity_ids).split(',')
    candidates = cand_map.get(s1_id, set())

    for true_id in true_ids:
        if true_id in candidates:
            continue

        s1_name = name_lookup.get(s1_id, '')
        match_name = name_lookup.get(true_id, '')
        s1_tokens = tokenize(s1_name)
        match_tokens = tokenize(match_name)
        shared = s1_tokens & match_tokens

        entry = (s1_id, true_id, s1_name, match_name)
        if len(shared) == 0:
            zero_token_misses.append(entry)
        else:
            had_token_misses.append(entry)

total_misses = len(zero_token_misses) + len(had_token_misses)
print(f"\nTotal missed true matches: {total_misses}", flush=True)
print(f"  Zero shared tokens (transliteration/typo case): {len(zero_token_misses)} ({100*len(zero_token_misses)/max(total_misses,1):.1f}%)", flush=True)
print(f"  Had shared tokens but still missed (cutoff/cap case): {len(had_token_misses)} ({100*len(had_token_misses)/max(total_misses,1):.1f}%)", flush=True)

print("\n--- Sample: ZERO shared tokens (need a second blocking signal) ---")
for s1_id, true_id, n1, n2 in zero_token_misses[:15]:
    print(f"  {s1_id} | {n1!r}  <->  {true_id} | {n2!r}")

print("\n--- Sample: HAD shared tokens but still missed (cutoff/cap issue) ---")
for s1_id, true_id, n1, n2 in had_token_misses[:15]:
    print(f"  {s1_id} | {n1!r}  <->  {true_id} | {n2!r}")

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)