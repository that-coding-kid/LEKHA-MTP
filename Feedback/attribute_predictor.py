# === Install and import libraries ===
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# === 1. Load HelpSteer dataset ===
ds = load_dataset("nvidia/HelpSteer")
train_ds = ds["train"] #35.8k samples
val_ds   = ds["validation"] #1.8k samples

# === 2. Initialize embedding models ===
# SBERT for sentence embeddings
sbert = SentenceTransformer('all-mpnet-base-v2')  # example SBERT model
sbert.to('cuda')

# SpanBERT encoder (use CLS token)
tokenizer_span = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
model_span     = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
model_span.to('cuda')

# Longformer encoder (use CLS token)
tokenizer_long = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
model_long     = AutoModel.from_pretrained("allenai/longformer-base-4096")
model_long.to('cuda')


# === 3. Define MLP for 5 attributes ===
# Determine embedding dimensions (SBERT:768, SpanBERT:768, Longformer:768)
input_dim = 768 + 768 +768  # Total: 2304
def make_mlp(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 2)
    ).to('cuda')

mlps_name = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

mlp_helpfulness = make_mlp(input_dim)
mlp_correctness = make_mlp(input_dim)
mlp_coherence   = make_mlp(input_dim)
mlp_complexity  = make_mlp(input_dim)
mlp_verbosity   = make_mlp(input_dim)

mlps = [mlp_helpfulness, mlp_correctness, mlp_coherence, mlp_complexity, mlp_verbosity]
# Loss and optimizer
criterion = nn.MSELoss()
optimizers = [torch.optim.Adam(mlp.parameters(), lr=1e-4) for mlp in mlps]

# === 4. Helper function: compute concatenated embedding ===
def embed_text(prompt, response):
    text = prompt + " [SEP] " + response
    
    # --- SBERT ---
    emb_sbert = sbert.encode(text, convert_to_tensor=True)  # shape (768,)
    if emb_sbert.dim() == 1:
        emb_sbert = emb_sbert.unsqueeze(0)  # (1,768)
    
    # --- SpanBERT (CLS token) ---
    inputs_span = tokenizer_span(text, truncation=True, max_length=512, return_tensors="pt").to('cuda')
    out_span = model_span(**inputs_span)
    emb_span = out_span.last_hidden_state[:, 0, :]  # (1,768)

    # --- Longformer (CLS token) ---
    inputs_long = tokenizer_long(text, truncation=True, max_length=4096, return_tensors="pt").to('cuda')
    out_long = model_long(**inputs_long)
    emb_long = out_long.last_hidden_state[:, 0, :]  # (1,768)
    
    # --- Concatenate along feature axis ---
    combined = torch.cat([emb_sbert, emb_span, emb_long], dim=1)  # (1,2304)
    return combined


# === 5. Prepare data loader ===
class HelpSteerDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        response = item["response"]
        # Each attribute as a float
        attrs = [
            torch.tensor(item["helpfulness"], dtype=torch.float),
            torch.tensor(item["correctness"], dtype=torch.float),
            torch.tensor(item["coherence"],   dtype=torch.float),
            torch.tensor(item["complexity"],  dtype=torch.float),
            torch.tensor(item["verbosity"],   dtype=torch.float)
        ]
        return prompt, response, attrs

train_data = HelpSteerDataset(train_ds)
val_data   = HelpSteerDataset(val_ds)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32)

# === 6. Training loop ===
for epoch in range(15):
    for mlp in mlps:
        mlp.train()
    total_losses = [0.0 for _ in range(5)]
    for prompts, responses, labels_list in tqdm(train_loader, desc=f"Epoch {epoch}"):
        batch_embeddings = []
        for p, r in zip(prompts, responses):
            emb = embed_text(p, r)
            batch_embeddings.append(emb)
        batch_embeddings = torch.cat(batch_embeddings, dim=0)  # (B, 2304)

        # Each attribute label: shape (B,)
        for i, (mlp, optimizer) in enumerate(zip(mlps, optimizers)):
            optimizer.zero_grad()
            labels = torch.stack([labels[i] for labels in labels_list]).to('cuda')  # (B,)
            # Expand labels to shape (B,2) for MSELoss (dummy second value, e.g. repeat)
            labels_2d = labels.unsqueeze(1).repeat(1,2)  # (B,2)
            preds = mlp(batch_embeddings)  # (B,2)
            loss = criterion(preds, labels_2d)
            loss.backward()
            optimizer.step()
            total_losses[i] += loss.item()
    print(f"Epoch {epoch}: Train losses = {[l/len(train_loader) for l in total_losses]}")

    # Validation
    for mlp in mlps:
        mlp.eval()
    val_losses = [0.0 for _ in range(5)]
    with torch.no_grad():
        for prompts, responses, labels_list in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
            batch_embeddings = []
            for p, r in zip(prompts, responses):
                emb = embed_text(p, r)
                batch_embeddings.append(emb)
            batch_embeddings = torch.cat(batch_embeddings, dim=0)
            for i, mlp in enumerate(mlps):
                labels = torch.stack([labels[i] for labels in labels_list]).to('cuda')
                labels_2d = labels.unsqueeze(1).repeat(1,2)
                preds = mlp(batch_embeddings)
                val_losses[i] += criterion(preds, labels_2d).item()
    print(f"Epoch {epoch}: Validation losses = {[l/len(val_loader) for l in val_losses]}")
    # Save model checkpoints
    for i, mlp in enumerate(mlps):
        torch.save(mlp.state_dict(), f"attribute_mlp_{mlps_name[i]}_epoch{epoch}.pth")