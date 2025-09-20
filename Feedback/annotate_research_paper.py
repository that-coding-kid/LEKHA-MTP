import json
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

data_path = "/home/teaching/yashasvi_mtp/LEKHA-MTP/Feedback/arxiv_dataset/arxiv-metadata-oai-snapshot.json"

records = []
with open(data_path, 'r') as f:
    for line in f:
        paper = json.loads(line)
        records.append({
            'id': paper['id'],
            'title': paper['title'],
            'abstract': paper['abstract'],
            'categories': paper['categories'],
            'doi': paper['doi'] if 'doi' in paper else None
        })

df = pd.DataFrame(records)
df.dropna(subset=['abstract'], inplace=True)

include_categories = ["cs.", "physics.", "math.", "q-bio.", "stat.", "econ.", "eess."]

# Filter dataset for abstracts that belong to selected domains
df = df[df['categories'].apply(lambda x: any(cat in x for cat in include_categories))]
df = df[df['abstract'].str.len() > 100]
df.drop_duplicates(subset=['abstract'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Sampling logic
samples_per_major = 7000
samples_per_minor = 5000
major_cats = ["cs.", "physics.", "math.", "q-bio.", "stat."]
minor_cats = ["econ.", "eess."]

sampled_dfs = []
for cat in include_categories:
    # Papers with this category
    cat_df = df[df['categories'].str.contains(cat)]
    n_samples = samples_per_major if cat in major_cats else samples_per_minor
    # If not enough papers, take all
    if len(cat_df) < n_samples:
        sampled = cat_df
    else:
        sampled = cat_df.sample(n=n_samples, random_state=42)
    sampled_dfs.append(sampled)

final_df = pd.concat(sampled_dfs).drop_duplicates(subset=['abstract']).reset_index(drop=True)
print(f"Total sampled papers: {len(final_df)}")
print(final_df['categories'].value_counts())
print(final_df.head())


# === 1. Load embedding models ===
sbert = SentenceTransformer('all-mpnet-base-v2', cache_folder="Sbert")
tokenizer_span = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
model_span     = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
tokenizer_long = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
model_long     = AutoModel.from_pretrained("allenai/longformer-base-4096")

def save_embeddings(model_name, model, tokenizer, dataframe, split):
    os.makedirs(f'embeddings/{model_name}/{split}', exist_ok=True)
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Embedding {model_name} {split}"):
        text = row['title'] + " [SEP] " + row['abstract']
        if model_name == "sbert":
            emb = model.encode(text, convert_to_tensor=True).cpu()
        else:
            inputs = tokenizer(text, truncation=True, max_length=(4096 if model_name=="longformer" else 512), return_tensors="pt")
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :].cpu().squeeze(0)
        torch.save(emb, f'embeddings/{model_name}/{split}/{idx}.pt')

# === 2. Store embeddings for all abstracts ===
save_embeddings("sbert", sbert, None, final_df, "arxiv_sample")
save_embeddings("spanbert", model_span, tokenizer_span, final_df, "arxiv_sample")
save_embeddings("longformer", model_long, tokenizer_long, final_df, "arxiv_sample")

# === 3. Predict attributes and add to dataframe ===
input_dim = 2304
def make_mlp(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 2)
    )
mlps_name = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
model_dir = "models"
epoch_to_test = 29
mlps = []
for name in mlps_name:
    mlp = make_mlp(input_dim)
    mlp.load_state_dict(torch.load(f"{model_dir}/attribute_mlp_{name}_epoch{epoch_to_test}.pth", map_location="cpu"))
    mlp.eval()
    mlps.append(mlp)

for i, name in enumerate(mlps_name):
    preds = []
    mlp = mlps[i]
    for idx in tqdm(range(len(final_df)), desc=f"Predicting {name}"):
        emb_sbert = torch.load(f'embeddings/sbert/arxiv_sample/{idx}.pt')
        emb_span  = torch.load(f'embeddings/spanbert/arxiv_sample/{idx}.pt')
        emb_long  = torch.load(f'embeddings/longformer/arxiv_sample/{idx}.pt')
        combined = torch.cat([emb_sbert, emb_span, emb_long], dim=0).unsqueeze(0)  # (1, 2304)
        with torch.no_grad():
            out = mlp(combined)
            pred = out[0, 0].item()
        preds.append(pred)
    final_df[f"pred_{name}"] = preds

final_df.to_csv("arxiv_sample_with_predictions.csv", index=False)
print("Saved to arxiv_sample_with_predictions.csv")