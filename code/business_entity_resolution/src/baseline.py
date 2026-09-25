import pandas as pd
import re
import time

def normalize_name(name):
    if pd.isna(name):
        return ""
    s = str(name).lower()
    s = re.sub(r'[^\w\s]', '', s)
    suffixes = ['inc', 'llc', 'ltd', 'limited', 'corp', 'corporation',
                'llp', 'pvt', 'private', 'co', 'company']
    tokens = [t for t in s.split() if t not in suffixes]
    return ' '.join(sorted(tokens))

def build_baseline(s1, s2, s3, max_group_size=15):
    s1 = s1.copy(); s2 = s2.copy(); s3 = s3.copy()
    s1['norm'] = s1['business_name'].apply(normalize_name)
    s2['norm'] = s2['business_name'].apply(normalize_name)
    s3['norm'] = s3['business_name'].apply(normalize_name)

    s2_idx = s2.groupby(['country', 'norm'])['entity_id'].apply(list)
    s2_idx = s2_idx[s2_idx.apply(len) <= max_group_size]

    s3_idx = s3.groupby(['country', 'norm'])['entity_id'].apply(list)
    s3_idx = s3_idx[s3_idx.apply(len) <= max_group_size]

    keys = list(zip(s1['country'], s1['norm']))

    m2 = [s2_idx.get(k, []) for k in keys]
    m3 = [s3_idx.get(k, []) for k in keys]

    matched_str = [','.join(a + b) for a, b in zip(m2, m3)]

    return pd.DataFrame({
        'source1_entity_id': s1['entity_id'].values,
        'matched_entity_ids': matched_str
    })

if __name__ == "__main__":
    t0 = time.time()
    s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t")
    s2 = pd.read_csv("dataset/test/test_source2.tsv", sep="\t")
    s3 = pd.read_csv("dataset/test/test_source3.tsv", sep="\t")
    print("data loaded", time.time()-t0)

    result = build_baseline(s1, s2, s3, max_group_size=8)
    print("baseline built", time.time()-t0)

    result.to_csv("output/matching_results.tsv", sep="\t", index=False)
    print("Done. Rows:", len(result))