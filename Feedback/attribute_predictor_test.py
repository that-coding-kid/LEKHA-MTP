import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset

# === 1. Load HelpSteer dataset ===
ds = load_dataset("nvidia/HelpSteer2")
test_ds   = ds["validation"]

# === 2. Precompute and store embeddings ===
def save_embeddings(model_name, model, tokenizer, dataset, split):
    os.makedirs(f'embeddings/{model_name}/{split}', exist_ok=True)
    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc=f"Embedding {model_name} {split}"):
        text = item["prompt"] + " [SEP] " + item["response"]
        if model_name == "sbert":
            emb = model.encode(text, convert_to_tensor=True).cpu()
        else:
            inputs = tokenizer(text, truncation=True, max_length=(4096 if model_name=="longformer" else 512), return_tensors="pt")
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :].cpu().squeeze(0)
        torch.save(emb, f'embeddings/{model_name}/{split}/{idx}.pt')

# Only run once to generate embeddings, then comment out
if not os.path.exists('embeddings/sbert/test/0.pt'):
    from sentence_transformers import SentenceTransformer
    print("Loading SBERT...")
    sbert = SentenceTransformer('all-mpnet-base-v2', cache_folder="Sbert")
    save_embeddings("sbert", sbert, None, test_ds, "test")
    del sbert
    torch.cuda.empty_cache()

if not os.path.exists('embeddings/spanbert/test/0.pt'):
    from transformers import AutoTokenizer, AutoModel
    print("Loading SpanBERT...")
    tokenizer_span = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
    model_span     = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
    save_embeddings("spanbert", model_span, tokenizer_span, test_ds, "test")
    del model_span, tokenizer_span
    torch.cuda.empty_cache()

if not os.path.exists('embeddings/longformer/test/0.pt'):
    from transformers import AutoTokenizer, AutoModel
    print("Loading Longformer...")
    tokenizer_long = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    model_long     = AutoModel.from_pretrained("allenai/longformer-base-4096")
    save_embeddings("longformer", model_long, tokenizer_long, test_ds, "test")
    del model_long, tokenizer_long
    torch.cuda.empty_cache()

# === 3. Dataset class using disk embeddings ===
class DiskEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, split, hf_dataset):
        self.split = split
        self.data = hf_dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        emb_sbert = torch.load(f'embeddings/sbert/{self.split}/{idx}.pt')
        emb_span  = torch.load(f'embeddings/spanbert/{self.split}/{idx}.pt')
        emb_long  = torch.load(f'embeddings/longformer/{self.split}/{idx}.pt')
        combined = torch.cat([emb_sbert, emb_span, emb_long], dim=0)  # (2304,)
        item = self.data[idx]
        attrs = [
            torch.tensor(item["helpfulness"], dtype=torch.float),
            torch.tensor(item["correctness"], dtype=torch.float),
            torch.tensor(item["coherence"],   dtype=torch.float),
            torch.tensor(item["complexity"],  dtype=torch.float),
            torch.tensor(item["verbosity"],   dtype=torch.float)
        ]
        return combined, attrs

test_dataset = DiskEmbeddingDataset("test", test_ds)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)


input_dim = 2304
def make_mlp(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 2)
    )

mlps_name = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

# === 2. Load models ===
model_dir = "models"
epoch_to_test = 29  # Change to the epoch you want to test
mlps = []
for name in mlps_name:
    mlp = make_mlp(input_dim)
    mlp.load_state_dict(torch.load(f"{model_dir}/attribute_mlp_{name}_epoch{epoch_to_test}.pth", map_location="cpu"))
    mlp.eval()
    mlps.append(mlp)

criterion = nn.MSELoss()

# === 3. Test loop ===
test_losses = [0.0 for _ in range(5)]
num_batches = 0

with torch.no_grad():
    for batch_embeddings, batch_labels in tqdm(test_loader, desc="Testing"):
        batch_attrs = torch.stack(batch_labels, dim=1)  # (B, 5)
        for i, mlp in enumerate(mlps):
            attr_vals = batch_attrs[:, i].float()  # (B,)
            # Scale attribute to [0, 1] across the batch (same as training)
            attr_min = attr_vals.min()
            attr_max = attr_vals.max()
            if attr_max - attr_min > 1e-8:
                scaled_attr = (attr_vals - attr_min) / (attr_max - attr_min)
            else:
                scaled_attr = attr_vals
            targets = torch.stack([scaled_attr, 1 - scaled_attr], dim=1)  # (B, 2)
            preds = mlp(batch_embeddings)
            loss = criterion(preds, targets)
            test_losses[i] += loss.item()
        num_batches += 1

print("Test MSE losses per attribute:")
for name, loss in zip(mlps_name, test_losses):
    print(f"{name}: {loss / num_batches:.4f}")