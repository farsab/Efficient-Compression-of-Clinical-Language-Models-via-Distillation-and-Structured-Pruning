import os
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import time

torch.manual_seed(42)
np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "processed_mimic10k"
MODEL_PATH = "teacher_model"
MAX_LEN = 512
BATCH_SIZE = 1
PRUNE_PCT = 0.1
EPOCHS = 1
LR = 1e-5
NUM_LABELS = 20
OUTPUT_DIR = "pruned_teacher_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MIMICDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

def get_model_size_mb(path):
    total = 0
    for f in os.listdir(path):
        total += os.path.getsize(os.path.join(path, f))
    return total / (1024 * 1024)

def load_pickle(name):
    with open(os.path.join(DATA_DIR, name), "rb") as f:
        return pickle.load(f)

# === Smart GRIFFIN for Prunning based on https://arxiv.org/abs/2402.19427===
def smart_griffin_ffn_slice(model, topk_ratio=0.9):
    for name, module in model.named_modules():
        if hasattr(module, 'intermediate') and hasattr(module, 'output'):
            linear1 = module.intermediate.dense
            linear2 = module.output.dense
            scores = linear1.weight.detach().pow(2).sum(dim=1)
            if scores.std() < 1e-3:
                print(f"Skipping {name} due to flat scores")
                continue
            k = max(1, int(scores.size(0) * topk_ratio))
            topk_indices = scores.topk(k).indices.sort().values
            with torch.no_grad():
                linear1.weight = torch.nn.Parameter(linear1.weight[topk_indices])
                linear1.bias = torch.nn.Parameter(linear1.bias[topk_indices])
                linear2.weight = torch.nn.Parameter(linear2.weight[:, topk_indices])
                linear1.out_features = k
                linear2.in_features = k
            print(f"Pruned {name} to {k} neurons")
    return model

def evaluate(model, dataloader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids=ids, attention_mask=attn).logits
            probas = torch.sigmoid(logits).cpu().numpy()
            preds.append(probas > 0.5)
            truths.append(labels)
    preds = np.vstack(preds)
    truths = np.vstack(truths)
    acc = (preds == truths).sum() / preds.size
    micro = f1_score(truths, preds, average="micro", zero_division=0)
    macro = f1_score(truths, preds, average="macro", zero_division=0)
    return acc, micro, macro

def train(model, train_dl, test_dl, device):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.to(device)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for step, batch in enumerate(progress, 1):
            opt.zero_grad()
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=ids, attention_mask=attn).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            progress.set_postfix(loss=running_loss / (step))

        acc_tr, micro_tr, macro_tr = evaluate(model, train_dl, device)
        print(f"Epoch {epoch+1} Train → Acc: {acc_tr:.4f}, Micro F1: {micro_tr:.4f}, Macro F1: {macro_tr:.4f}")

        if epoch % 5 == 0:
            acc_te, micro_te, macro_te = evaluate(model, test_dl, device)
            print(f"Epoch {epoch+1} Test  → Acc: {acc_te:.4f}, Micro F1: {micro_te:.4f}, Macro F1: {macro_te:.4f}")

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency(model, dataloader, device, num_batches=20):
    model.eval()
    times = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            start = time.time()
            _ = model(input_ids=input_ids, attention_mask=attn).logits
            end = time.time()
            times.append((end - start) * 1000)
    return np.mean(times)

X_train = load_pickle("X_train.pkl")
X_test = load_pickle("X_test.pkl")
y_train = load_pickle("y_train.pkl")
y_test = load_pickle("y_test.pkl")
mlb = load_pickle("label_binarizer.pkl")

label_counts = y_train.sum(axis=0)
top_20_indices = label_counts.argsort()[::-1][:20]
top_20_labels = np.array(mlb.classes_)[top_20_indices]
y_train = y_train[:, top_20_indices]
y_test = y_test[:, top_20_indices]

valid_indices = y_train.sum(axis=1) > 0
X_train = [X_train[i] for i in range(len(X_train)) if valid_indices[i]]
y_train = y_train[valid_indices]
valid_indices_test = y_test.sum(axis=1) > 0
X_test = [X_test[i] for i in range(len(X_test)) if valid_indices_test[i]]
y_test = y_test[valid_indices_test]
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

train_dl = DataLoader(MIMICDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(MIMICDataset(X_test, y_test), batch_size=1)

original_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

print("Evaluating original model...")
original_metrics = evaluate(original_model, test_dl, device)
original_size = count_parameters(original_model)
latency_orig = measure_latency(original_model, test_dl, device)
print("Before Pruning: (Accuracy, Micro F1, Macro F1)", original_metrics)

print("Applying Smart GRIFFIN pruning...")
model = smart_griffin_ffn_slice(original_model, topk_ratio=1 - PRUNE_PCT)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Retraining pruned model...")
model = train(model, train_dl, test_dl, device)

print("Final evaluation on test set:")
final_metrics = evaluate(model, test_dl, device)
pruned_size = count_parameters(model)
latency_pruned = measure_latency(model, test_dl, device)
print("After Pruning:", final_metrics)
size_on_disk_orig = get_model_size_mb(MODEL_PATH)
size_on_disk_pruned = get_model_size_mb(OUTPUT_DIR)

df = pd.DataFrame([{
    "Model": "Original",
    "Params": original_size,
    "Size_MB": size_on_disk_orig,  
    "Latency_ms": latency_orig,
    "Acc": original_metrics[0],
    "Micro_F1": original_metrics[1],
    "Macro_F1": original_metrics[2],
    "Compression_Ratio": 1.0,
    "Speedup": 1.0
}, {
    "Model": "Pruned",
    "Params": pruned_size,
    "Size_MB": size_on_disk_pruned, 
    "Latency_ms": latency_pruned,
    "Acc": final_metrics[0],
    "Micro_F1": final_metrics[1],
    "Macro_F1": final_metrics[2],
    "Compression_Ratio": original_size / pruned_size,
    "Speedup": latency_orig / latency_pruned

}])

print(df.to_string(index=False))

df.to_csv(os.path.join(SAVE_DIR,"pruned_teacher_griffin_comparison.csv"), index=False)
print("\nSaved to griffin_comparison.csv")
