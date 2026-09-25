import pandas as pd
import re
import time
import gc

STOPWORDS = {'inc', 'llc', 'ltd', 'limited', 'corp', 'corporation',
             'llp', 'pvt', 'private', 'co', 'company', 'the', 'and'}

def tokenize(name):
    if pd.isna(name):
        return []
    s = str(name).lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    return [t for t in s.split() if t not in STOPWORDS and len(t) > 1]

def explode_tokens(df, id_col='entity_id'):
    tmp = df[[id_col, 'country', 'tokens']].explode('tokens').dropna(subset=['tokens'])
    tmp = tmp.rename(columns={'tokens': 'token'})
    return tmp

def cap_postings(tok_df, col='token', max_postings=300):
    counts = tok_df[col].value_counts()
    keep = counts[counts <= max_postings].index
    return tok_df[tok_df[col].isin(keep)]

def match_and_rank(s1_tok, src_df, label, max_postings=300, max_candidates=40, chunk_size=5000):
    src = src_df.copy()
    src['tokens'] = src['business_name'].apply(tokenize)
    src_tok = explode_tokens(src, 'entity_id').rename(columns={'entity_id': 'cand_id', 'country': 'cand_country'})
    del src
    gc.collect()

    src_tok = cap_postings(src_tok, max_postings=max_postings)

    all_counts = []
    countries = s1_tok['s1_country'].dropna().unique()
    for country in countries:
        left_country = s1_tok[s1_tok['s1_country'] == country]
        right_country = src_tok[src_tok['cand_country'] == country]
        if left_country.empty or right_country.empty:
            continue

        s1_ids_in_country = left_country['s1_id'].unique()

        for i in range(0, len(s1_ids_in_country), chunk_size):
            batch_ids = s1_ids_in_country[i:i+chunk_size]
            left_batch = left_country[left_country['s1_id'].isin(batch_ids)]

            merged = left_batch.merge(right_country, on='token')
            if not merged.empty:
                counts = merged.groupby(['s1_id', 'cand_id']).size().reset_index(name='shared')
                counts = counts.sort_values(['s1_id', 'shared'], ascending=[True, False])
                counts['rank'] = counts.groupby('s1_id').cumcount()
                counts = counts[counts['rank'] < max_candidates]
                all_counts.append(counts[['s1_id', 'cand_id']])
            del merged
            gc.collect()

        print(f"  [{label}] done country={country}", flush=True)

    del src_tok
    gc.collect()

    if not all_counts:
        return pd.Series(dtype=object)
    counts = pd.concat(all_counts, ignore_index=True)
    return counts.groupby('s1_id')['cand_id'].apply(list)

def build_candidates(s1, s2, s3, max_postings=300, max_candidates=40, chunk_size=5000):
    s1 = s1.copy()
    s1['tokens'] = s1['business_name'].apply(tokenize)
    s1_tok = explode_tokens(s1, 'entity_id').rename(columns={'entity_id': 's1_id', 'country': 's1_country'})
    s1_tok = cap_postings(s1_tok, max_postings=max_postings)

    print("processing s2...", flush=True)
    c2 = match_and_rank(s1_tok, s2, 's2', max_postings, max_candidates, chunk_size)

    print("processing s3...", flush=True)
    c3 = match_and_rank(s1_tok, s3, 's3', max_postings, max_candidates, chunk_size)

    result = pd.DataFrame({'source1_entity_id': s1['entity_id']})
    result = result.merge(c2.rename('c2'), left_on='source1_entity_id', right_index=True, how='left')
    result = result.merge(c3.rename('c3'), left_on='source1_entity_id', right_index=True, how='left')
    result['c2'] = result['c2'].apply(lambda x: x if isinstance(x, list) else [])
    result['c3'] = result['c3'].apply(lambda x: x if isinstance(x, list) else [])
    result['candidate_entity_ids'] = (result['c2'] + result['c3']).apply(lambda x: ','.join(x))
    return result[['source1_entity_id', 'candidate_entity_ids']]

if __name__ == "__main__":
    t0 = time.time()
    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
    s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
    s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
    print("loaded", time.time()-t0, flush=True)

    result = build_candidates(s1, s2, s3, max_postings=300, max_candidates=40, chunk_size=5000)
    print("built", time.time()-t0, flush=True)

    result.to_csv("output/candidate_pairs.tsv", sep="\t", index=False)
    print("Done. Rows:", len(result), time.time()-t0)