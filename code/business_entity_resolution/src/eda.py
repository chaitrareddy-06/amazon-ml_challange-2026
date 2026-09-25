import pandas as pd

s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

print("=== Row counts ===")
print(f"Source1: {len(s1)}, Source2: {len(s2)}, Source3: {len(s3)}, Ground truth: {len(gt)}")

print("\n=== Country distribution ===")
print("S1:", s1['country'].value_counts().to_dict())
print("S2:", s2['country'].value_counts().to_dict())
print("S3:", s3['country'].value_counts().to_dict())

print("\n=== Singleton rate (critical for F0.5) ===")
gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
gt['n_matches'] = gt['matched_entity_ids'].apply(lambda x: 0 if x == '' else len(str(x).split(',')))
singleton_rate = (gt['n_matches'] == 0).mean()
print(f"Singleton rate: {singleton_rate:.2%}")
print("Match count distribution:", gt['n_matches'].value_counts().sort_index().to_dict())

print("\n=== Sample name variations (same S1 entity, its matches) ===")
sample = gt[gt['n_matches'] > 0].head(5)
for _, row in sample.iterrows():
    s1_name = s1.loc[s1['entity_id'] == row['source1_entity_id'], 'business_name'].values
    print(f"\nS1 ({row['source1_entity_id']}): {s1_name[0] if len(s1_name) else '?'}")
    for mid in str(row['matched_entity_ids']).split(','):
        src = s2 if mid.startswith('S2-') else s3
        name = src.loc[src['entity_id'] == mid, 'business_name'].values
        addr = src.loc[src['entity_id'] == mid, 'business_address'].values
        print(f"  {mid}: {name[0] if len(name) else '?'} | {addr[0] if len(addr) else '?'}")

print("\n=== Missing values ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print(f"{name}:", df.isnull().sum().to_dict())