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
# Simple 2-layer MLP
mlp = nn.Sequential(
    nn.Linear(input_dim, 512),
    nn.ReLU(),
    nn.Linear(512, 5)
).to('cuda')

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

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
        # Attributes as a tensor of shape (5,)
        attrs = torch.tensor([item["helpfulness"],
                               item["correctness"],
                               item["coherence"],
                               item["complexity"],
                               item["verbosity"]],
                              dtype=torch.float)
        return prompt, response, attrs

train_data = HelpSteerDataset(train_ds)
val_data   = HelpSteerDataset(val_ds)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32)

# === 6. Training loop ===
for epoch in range(15):  # example: 3 epochs
    mlp.train()
    total_loss = 0.0
    for prompts, responses, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        batch_embeddings = []
        # Compute embeddings for each example in batch
        for p, r in zip(prompts, responses):
            emb = embed_text(p, r)     # (1,2304)
            batch_embeddings.append(emb)
        batch_embeddings = torch.cat(batch_embeddings, dim=0)  # (B, 2304)
        labels = labels.to('cuda')  # (B,5)
        preds = mlp(batch_embeddings)  # (B,5)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Train loss = {total_loss/len(train_loader):.4f}")

    # (Optional) Evaluate on validation set similarly
    mlp.eval()
    val_loss = 0.0
    with torch.no_grad():
        for prompts, responses, labels in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
            batch_embeddings = []
            for p, r in zip(prompts, responses):
                emb = embed_text(p, r)
                batch_embeddings.append(emb)
            batch_embeddings = torch.cat(batch_embeddings, dim=0)
            labels = labels.to('cuda')
            preds = mlp(batch_embeddings)
            val_loss += criterion(preds, labels).item()
    print(f"Epoch {epoch}: Validation loss = {val_loss/len(val_loader):.4f}")
    # Save model checkpoint
    torch.save(mlp.state_dict(), f"attribute_mlp_epoch{epoch}.pth")