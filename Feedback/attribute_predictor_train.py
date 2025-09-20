import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset

# === 1. Load HelpSteer dataset ===
ds = load_dataset("nvidia/HelpSteer")
train_ds = ds["train"]
val_ds   = ds["validation"]

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
if not os.path.exists('embeddings/sbert/train/0.pt'):
    from sentence_transformers import SentenceTransformer
    print("Loading SBERT...")
    sbert = SentenceTransformer('all-mpnet-base-v2', cache_folder="Sbert")
    save_embeddings("sbert", sbert, None, train_ds, "train")
    save_embeddings("sbert", sbert, None, val_ds, "val")
    del sbert
    torch.cuda.empty_cache()

if not os.path.exists('embeddings/spanbert/train/0.pt'):
    from transformers import AutoTokenizer, AutoModel
    print("Loading SpanBERT...")
    tokenizer_span = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
    model_span     = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
    save_embeddings("spanbert", model_span, tokenizer_span, train_ds, "train")
    save_embeddings("spanbert", model_span, tokenizer_span, val_ds, "val")
    del model_span, tokenizer_span
    torch.cuda.empty_cache()

if not os.path.exists('embeddings/longformer/train/0.pt'):
    from transformers import AutoTokenizer, AutoModel
    print("Loading Longformer...")
    tokenizer_long = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    model_long     = AutoModel.from_pretrained("allenai/longformer-base-4096")
    save_embeddings("longformer", model_long, tokenizer_long, train_ds, "train")
    save_embeddings("longformer", model_long, tokenizer_long, val_ds, "val")
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

train_data = DiskEmbeddingDataset("train", train_ds)
val_data   = DiskEmbeddingDataset("val", val_ds)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=16)

# === 4. Define MLPs ===
input_dim = 2304
def make_mlp(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 2)
    )

mlps_name = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
mlps = [make_mlp(input_dim) for _ in range(5)]
criterion = nn.MSELoss()
optimizers = [torch.optim.Adam(mlp.parameters(), lr=1e-4) for mlp in mlps]

# === 5. Training loop with load/deload ===
for epoch in range(30):
    total_losses = [0.0 for _ in range(5)]
    for batch_embeddings, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        batch_embeddings = batch_embeddings  # (B, 2304)
        
        # Debug prints for first batch
        if epoch == 0 and 'debug_printed' not in locals():
            print(f"Type of batch_labels: {type(batch_labels)}")
            print(f"Length of batch_labels: {len(batch_labels)}")
            print(f"Type of first element: {type(batch_labels[0])}")
            print(f"Shape of first element: {batch_labels[0].shape if hasattr(batch_labels[0], 'shape') else 'No shape'}")
            debug_printed = True
        
        # Handle batch_labels properly - DataLoader returns list of attribute tensors
        # batch_labels is a list of 5 tensors, each of size (B,)
        batch_attrs = torch.stack(batch_labels, dim=1)  # Stack along dim 1 to get (B, 5)
        
        for i, (mlp, optimizer) in enumerate(zip(mlps, optimizers)):
            mlp = mlp.to('cuda')
            optimizer.zero_grad()
            
            # Get the i-th attribute for all samples in the batch
            attr_vals = batch_attrs[:, i].float().to('cuda')  # (B,)
            
            # Scale attribute to [0, 1] across the batch
            attr_min = attr_vals.min()
            attr_max = attr_vals.max()
            if attr_max - attr_min > 1e-8:
                scaled_attr = (attr_vals - attr_min) / (attr_max - attr_min)
            else:
                scaled_attr = attr_vals  # If all values are the same, keep them as is
            
            targets = torch.stack([scaled_attr, 1 - scaled_attr], dim=1)  # shape: [B, 2]
            preds = mlp(batch_embeddings.to('cuda'))
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_losses[i] += loss.item()
            mlp = mlp.to('cpu')
            torch.cuda.empty_cache()
            mlps[i] = mlp
            
    print(f"Epoch {epoch}: Train losses = {[l/len(train_loader) for l in total_losses]}")

    # Validation
    val_losses = [0.0 for _ in range(5)]
    with torch.no_grad():
        for batch_embeddings, batch_labels in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
            # Handle batch_labels - list of 5 tensors, each of size (B,)
            batch_attrs = torch.stack(batch_labels, dim=1)  # Stack along dim 1 to get (B, 5)
            
            for i, mlp in enumerate(mlps):
                mlp = mlp.to('cuda')
                
                # Get the i-th attribute for all samples in the batch
                attr_vals = batch_attrs[:, i].float().to('cuda')  # (B,)
                
                # Scale attribute to [0, 1] across the batch
                attr_min = attr_vals.min()
                attr_max = attr_vals.max()
                if attr_max - attr_min > 1e-8:
                    scaled_attr = (attr_vals - attr_min) / (attr_max - attr_min)
                else:
                    scaled_attr = attr_vals  # If all values are the same, keep them as is
                
                targets = torch.stack([scaled_attr, 1 - scaled_attr], dim=1)  # shape: [B, 2]
                preds = mlp(batch_embeddings.to('cuda'))
                val_losses[i] += criterion(preds, targets).item()
                mlp = mlp.to('cpu')
                torch.cuda.empty_cache()
                mlps[i] = mlp
                
    print(f"Epoch {epoch}: Validation losses = {[l/len(val_loader) for l in val_losses]}")
    if epoch % 10 == 9:
        for i, mlp in enumerate(mlps):
            torch.save(mlp.state_dict(), f"models/attribute_mlp_{mlps_name[i]}_epoch{epoch}.pth")