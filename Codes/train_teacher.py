import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

torch.manual_seed(42)
np.random.seed(42)
DATA_DIR = "processed_mimic10k"
MODEL_NAME = "michiyasunaga/BioLinkBERT-large"
MAX_LEN = 512
BATCH_SIZE = 1
EPOCHS = 50
LR = 1e-5
PATIENCE = 10
NUM_LABELS = 20
OUTPUT_DIR = "teacher_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_DIR = "teacher_metrics"
os.makedirs(SAVE_DIR, exist_ok=True)

class MIMICDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.FloatTensor(self.labels[idx])
        return item

def compute_accuracy(y_true, y_pred):
    return (y_true == y_pred).sum() / np.prod(y_true.shape)

def evaluate(model, dataloader, name=""):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probas = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probas > 0.5)
            truths.extend(labels)

    preds = np.array(preds)
    truths = np.array(truths)
    acc = compute_accuracy(truths, preds)
    micro = f1_score(truths, preds, average="micro", zero_division=0)
    macro = f1_score(truths, preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(truths, preds, average=None, zero_division=0)

    print(f"{name} Acc: {acc:.4f}, Micro F1: {micro:.4f}, Macro F1: {macro:.4f}")
    return acc, micro, macro, per_class_f1

def load_pickle(name):
    with open(os.path.join(DATA_DIR, name), "rb") as f:
        return pickle.load(f)

train_losses, val_losses = [], []
train_micro_f1s, val_micro_f1s = [], []
train_macro_f1s, val_macro_f1s = [], []
train_accuracies, val_accuracies = [], []
test_accuracies, test_micro_f1s, test_macro_f1s = [], [], []

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

valid_indices = (y_train.sum(axis=1) > 0)
X_train = [X_train[i] for i in range(len(X_train)) if valid_indices[i]]
y_train = y_train[valid_indices]

# Build a map of each label to the list of samples that have it
label_to_samples = defaultdict(list)
for i in range(len(y_train)):
    for label_idx in np.where(y_train[i] == 1)[0]:
        label_to_samples[label_idx].append(i)

if not label_to_samples:
    target_count = 0
else:
    for i in range(y_train.shape[1]):
        if i not in label_to_samples:
            label_to_samples[i] = []
    target_count = min(len(samples) for samples in label_to_samples.values())

print(f"\nTarget count per class (upper limit): {target_count}")

rng = np.random.default_rng(42)
for label in label_to_samples:
    rng.shuffle(label_to_samples[label])

# Iterative Balancing Algorithm using Round-robin technique
final_indices = set()
final_counts = np.zeros(y_train.shape[1], dtype=int)
sample_pointers = defaultdict(int)

progress_made = True
while progress_made:
    progress_made = False

    for label_idx in range(y_train.shape[1]):
        if final_counts[label_idx] >= target_count:
            continue

        while sample_pointers[label_idx] < len(label_to_samples[label_idx]):
            sample_idx = label_to_samples[label_idx][sample_pointers[label_idx]]
            sample_pointers[label_idx] += 1  

            if sample_idx in final_indices:
                continue 

            sample_labels_vector = y_train[sample_idx]
            potential_counts = final_counts + sample_labels_vector

            if np.all(potential_counts <= target_count):
                final_indices.add(sample_idx)
                final_counts = potential_counts
                progress_made = True
                break 

final_indices_list = sorted(list(final_indices))

X_train = [X_train[i] for i in final_indices_list]
y_train = y_train[final_indices_list]

counts = y_train.sum(axis=0).astype(int)
print("\nFinal Label Counts:")
df = pd.DataFrame({
    "Label Index": np.arange(20),
    "Label Name": top_20_labels,
    "Count": counts
})
print(df.to_string(index=False))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
train_dl = DataLoader(MIMICDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(MIMICDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_dl = DataLoader(MIMICDataset(X_test, y_test), batch_size=BATCH_SIZE)

print("Dataset Summary")
print(f" - Train Samples: {len(X_tr)}")
print(f" - Val Samples: {len(X_val)}")
print(f" - Test Samples: {len(X_test)}")
print(f" - Number of Classes: {NUM_LABELS}")
print(f" - Model: {MODEL_NAME}")
print(f" - Max Length: {MAX_LEN}")
print(f" - Batch Size: {BATCH_SIZE}")
print(f" - Epochs: {EPOCHS}")
print(f" - Learning Rate: {LR}")
print(f" - Patience (Early Stop): {PATIENCE}")
print("-" * 40)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS, problem_type="multi_label_classification"
).cuda()

optimizer = AdamW(model.parameters(), lr=LR)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dl)*EPOCHS)

best_val_loss = float("inf")
patience_counter = 0

per_class_f1_df = pd.DataFrame(columns=["epoch"] + list(top_20_labels))
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    train_loss = 0
    train_preds, train_truths = [], []

    for batch in tqdm(train_dl):
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()

        probas = torch.sigmoid(outputs.logits).detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        train_preds.extend(probas > 0.5)
        train_truths.extend(labels)

    avg_train_loss = train_loss / len(train_dl)
    train_acc = compute_accuracy(np.array(train_truths), np.array(train_preds))
    train_micro = f1_score(train_truths, train_preds, average="micro")
    train_macro = f1_score(train_truths, train_preds, average="macro")

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)
    train_micro_f1s.append(train_micro)
    train_macro_f1s.append(train_macro)

    print(f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, Micro F1: {train_micro:.4f}, Macro F1: {train_macro:.4f}")

    val_loss = 0
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_dl)
    val_losses.append(avg_val_loss)
    val_acc, val_micro, val_macro, val_per_class_f1 = evaluate(model, val_dl, name="Val")
    val_accuracies.append(val_acc)
    val_micro_f1s.append(val_micro)
    val_macro_f1s.append(val_macro)
    test_acc, test_micro, test_macro, _ = evaluate(model, test_dl, name="Test")
    test_accuracies.append(test_acc)
    test_micro_f1s.append(test_micro)
    test_macro_f1s.append(test_macro)
    per_class_f1_df.loc[len(per_class_f1_df)] = [epoch + 1] + list(val_per_class_f1)

    # EARLY STOP MECHANISM
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print("Best model saved.")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    else:
        patience_counter += 1
        print(f"Early Stop Counter: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

metrics_df = pd.DataFrame({
    "epoch": np.arange(1, len(train_losses) + 1),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_acc": train_accuracies,
    "val_acc": val_accuracies,
    "test_acc": test_accuracies,
    "train_micro_f1": train_micro_f1s,
    "val_micro_f1": val_micro_f1s,
    "test_micro_f1": test_micro_f1s,
    "train_macro_f1": train_macro_f1s,
    "val_macro_f1": val_macro_f1s,
    "test_macro_f1": test_macro_f1s
})
metrics_df.to_csv(os.path.join(SAVE_DIR, "teacher_training_metrics.csv"), index=False)
per_class_f1_df.to_csv(os.path.join(SAVE_DIR, "teacher_val_per_class_f1.csv"), index=False)

print("\nFinal Test Evaluation:")
test_acc, test_micro, test_macro, test_per_class_f1 = evaluate(model, test_dl, name="Test")

test_metrics_df = pd.DataFrame({
    "label": top_20_labels,
    "f1_score": test_per_class_f1
})
test_metrics_df.to_csv(os.path.join(SAVE_DIR, "teacher_test_per_class_f1.csv"), index=False)

with open(os.path.join(SAVE_DIR, "test_summary.csv"), "w", newline='') as f:
    f.write("acc,micro_f1,macro_f1\n")
    f.write(f"{test_acc:.4f},{test_micro:.4f},{test_macro:.4f}\n")

plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(train_micro_f1s, label="Train Micro F1")
plt.plot(val_micro_f1s, label="Val Micro F1")
plt.plot(train_macro_f1s, label="Train Macro F1")
plt.plot(val_macro_f1s, label="Val Macro F1")
plt.title("F1 per Epoch")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "teacher_training_curves.png"))
