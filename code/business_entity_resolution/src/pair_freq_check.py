import pandas as pd
import re
import time
from itertools import combinations
from collections import Counter

STOPWORDS = {'inc', 'llc', 'ltd', 'limited', 'corp', 'corporation',
             'llp', 'pvt', 'private', 'co', 'company', 'the', 'and'}

def tokenize(name):
    if pd.isna(name):
        return []
    s = str(name).lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    return [t for t in s.split() if t not in STOPWORDS and len(t) > 1]

def make_pair_tokens(tokens, max_pairs=6):
    toks = sorted(set(tokens))[:6]
    return ['|'.join(p) for p in combinations(toks, 2)][:max_pairs]

t0 = time.time()
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
print(f"loaded {time.time()-t0:.1f}s", flush=True)

pair_freq = Counter()
for name in pd.concat([s2['business_name'], s3['business_name']]):
    toks = tokenize(name)
    for pair in make_pair_tokens(toks):
        pair_freq[pair] += 1
print(f"pair freq built {time.time()-t0:.1f}s, unique pairs={len(pair_freq)}", flush=True)

# Check the specific pairs relevant to your earlier miss samples
check_pairs = ['raj|investments', 'red|ventures', 'private|ventures', 'investments|llp']
for p in check_pairs:
    print(f"  '{p}': {pair_freq.get(p, 0)} postings", flush=True)

over_cap = sum(1 for c in pair_freq.values() if c > 300)
total_postings = sum(pair_freq.values())
dropped = sum(c for c in pair_freq.values() if c > 300)
print(f"\nPair-tokens over cap (300): {over_cap} / {len(pair_freq)} ({100*over_cap/len(pair_freq):.2f}%)", flush=True)
print(f"Postings dropped by cap: {dropped} / {total_postings} ({100*dropped/max(total_postings,1):.2f}%)", flush=True)
print(f"Total time: {time.time()-t0:.1f}s", flush=True)