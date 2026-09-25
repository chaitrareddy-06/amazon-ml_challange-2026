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
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
print(f"loaded {time.time()-t0:.1f}s", flush=True)

from collections import Counter
token_freq = Counter()
for name in pd.concat([s2['business_name'], s3['business_name']]):
    for tok in tokenize(name):
        token_freq[tok] += 1
print(f"token freq built {time.time()-t0:.1f}s, unique tokens={len(token_freq)}", flush=True)

# Check the specific tokens from your miss samples
check_tokens = ['payne', 'raj', 'red', 'investments', 'ventures', 'lumay', 'boral', 'dahlia']
for t in check_tokens:
    print(f"  '{t}': {token_freq.get(t, 0)} postings", flush=True)

# What % of ALL tokens exceed the current cap of 300?
over_cap = sum(1 for c in token_freq.values() if c > 300)
print(f"\nTokens over cap (300): {over_cap} / {len(token_freq)} ({100*over_cap/len(token_freq):.2f}%)", flush=True)

# More importantly: what % of TOTAL POSTINGS (not unique tokens) get dropped by the cap?
total_postings = sum(token_freq.values())
dropped_postings = sum(c for c in token_freq.values() if c > 300)
print(f"Postings dropped by cap: {dropped_postings} / {total_postings} ({100*dropped_postings/total_postings:.2f}%)", flush=True)