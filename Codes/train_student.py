import pandas as pd
import os, csv, pickle, torch
import numpy as np
from tqdm import tqdm
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
import time


torch.manual_seed(42)
np.random.seed(42)
DATA_DIR = "processed_mimic10k"
TEACHER_PATH = "teacher_model"
STUDENT_MODEL_NAME = "michiyasunaga/BioLinkBERT-base"
BATCH_SIZE = 1
EPOCHS = 50
LR = 1e-5
PATIENCE = 5
NUM_LABELS = 20
OUTPUT_DIR = "student_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_DIR = "student_metrics"
os.makedirs(SAVE_DIR, exist_ok=True)

METRIC_FILE = r"./student_metrics/student_metrics_summary.csv"

# Expand these to search for best values
ALPHA_VALUES = [0.3] 
TEMP_VALUES = [1.0]

class MIMICDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.FloatTensor(self.labels[idx])
        return item
def print_model_size(model, name="Model"):
    temp_path = f"{name}_temp_size"
    model.save_pretrained(temp_path)
    total_size = sum(os.path.getsize(os.path.join(temp_path, f)) for f in os.listdir(temp_path)) / (1024 * 1024)
    print(f"{name} size: {total_size:.2f} MB")
    import shutil
    shutil.rmtree(temp_path)

def print_model_parameters(model, name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name} Parameters:")
    print(f"   - Total: {total_params:,}")
    print(f"   - Trainable: {trainable_params:,}")
    
def compute_and_log_metrics(epoch, split, preds, truths, loss, path=METRIC_FILE, return_scores=False):
    preds_bin = (preds > 0.5).astype(int)
    micro = f1_score(truths, preds_bin, average="micro")
    macro = f1_score(truths, preds_bin, average="macro")
    acc = accuracy_score(truths.flatten(), preds_bin.flatten())
    per_class_f1 = f1_score(truths, preds_bin, average=None).tolist()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, split, loss, micro, macro, acc] + per_class_f1)

    print(f"[{split}] Loss: {loss:.4f} | Micro F1: {micro:.4f} | Macro F1: {macro:.4f} | Acc: {acc:.4f}")
    if return_scores:
        return loss, micro, macro, acc, per_class_f1

def load_pickle(name):
    with open(os.path.join(DATA_DIR, name), "rb") as f:
        return pickle.load(f)

def plot_metric(df,metric, title, ylabel, filename):
    plt.figure(figsize=(10, 5))
    for split in ["train", "test"]:
        sub = df[df["split"] == split]
        if not sub.empty:
            plt.plot(sub["epoch"], sub[metric], label=split)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()

def evaluate_model(model, dataloader, device="cuda"):
    model.eval()
    all_preds, all_truths = [], []
    start = time.time()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            truths = batch["labels"].cpu().numpy()
            all_preds.extend(probs)
            all_truths.extend(truths)
    end = time.time()

    latency_ms = (end - start) / len(dataloader.dataset) * 1000
    preds_bin = (np.array(all_preds) > 0.5).astype(int)
    truths = np.array(all_truths)

    acc = accuracy_score(truths.flatten(), preds_bin.flatten())
    micro = f1_score(truths, preds_bin, average="micro")
    macro = f1_score(truths, preds_bin, average="macro")

    return latency_ms, acc, micro, macro
def get_model_stats(model, name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    tmp_dir = f"./tmp_{name}"
    model.save_pretrained(tmp_dir)
    size_mb = sum(os.path.getsize(os.path.join(tmp_dir, f)) for f in os.listdir(tmp_dir)) / (1024 * 1024)
    import shutil; shutil.rmtree(tmp_dir)
    return total_params, size_mb

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

# Remove train samples with no top-20 labels
valid_train = (y_train.sum(axis=1) > 0)
X_train = [X_train[i] for i in range(len(X_train)) if valid_train[i]]
y_train = y_train[valid_train]

valid_test = (y_test.sum(axis=1) > 0)
X_test = [X_test[i] for i in range(len(X_test)) if valid_test[i]]
y_test = y_test[valid_test]

CLASSES = top_20_labels

# Build a map of each label to the list of samples that have it
label_to_samples = defaultdict(list)
for i in range(len(y_train)):
    for label_idx in np.where(y_train[i] == 1)[0]:
        label_to_samples[label_idx].append(i)

for i in range(y_train.shape[1]):
    if i not in label_to_samples:
        label_to_samples[i] = []

target_count = min(len(samples) for samples in label_to_samples.values())
print(f"\nTarget count per class (upper limit): {target_count}")

rng = np.random.default_rng(42)
for label in label_to_samples:
    rng.shuffle(label_to_samples[label])

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
                continue  # Already used

            sample_labels = y_train[sample_idx]
            potential_counts = final_counts + sample_labels

            if np.all(potential_counts <= target_count):
                final_indices.add(sample_idx)
                final_counts = potential_counts
                progress_made = True
                break  # Go to next label
final_indices = sorted(list(final_indices))
X_train = [X_train[i] for i in final_indices]
y_train = y_train[final_indices]

counts = y_train.sum(axis=0).astype(int)
print("\nFinal Label Counts:")
df = pd.DataFrame({
    "Label Index": np.arange(NUM_LABELS),
    "Label Name": CLASSES,
    "Count": counts
})

X_tr = X_train
y_tr = y_train
print("Dataset Summary")
print(f" - Train: {len(X_tr)} | Test: {len(X_test)}")
print(f" - Num Classes: {NUM_LABELS}")
print(f" - Student: {STUDENT_MODEL_NAME}")
print(f" - Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LR}")
print("-" * 40)
label_distribution = np.sum(y_train, axis=0)
print("Label distribution (train):", label_distribution)
print("Train label counts:", y_train.sum(axis=0))
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)

train_dl = DataLoader(MIMICDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(MIMICDataset(X_test, y_test), batch_size=BATCH_SIZE)

teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_PATH).cuda().eval()
results = []
print_model_size(teacher, name="Teacher")
print_model_parameters(teacher, name="Teacher")

for alpha, temp in product(ALPHA_VALUES, TEMP_VALUES):
    print(f"\nTrying α={alpha}, T={temp}")

    student = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL_NAME, num_labels=NUM_LABELS, problem_type="multi_label_classification"
    ).cuda()
    print_model_size(student, name="Student")
    print_model_parameters(student, name="Student")
    optimizer = AdamW(student.parameters(), lr=LR)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dl) * EPOCHS)
    best_val_f1 = 0
    patience_counter = 0

    with open(METRIC_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "split", "loss", "micro_f1", "macro_f1", "accuracy"] + list(CLASSES))

    best_val_loss = float("inf")
    patience_counter = 0
    best_test_acc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        student.train()
        total_loss = 0
        all_preds, all_truths = [], []

        for batch in tqdm(train_dl):
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                teacher_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

            student_outputs = student(**batch)
            if epoch == 0 and len(all_preds) == 0:
                probs = torch.sigmoid(student_outputs.logits).detach().cpu().numpy()

            hard_loss = F.binary_cross_entropy_with_logits(student_outputs.logits, batch["labels"])
            soft_loss = F.kl_div(
                F.logsigmoid(student_outputs.logits / temp),
                F.sigmoid(teacher_logits / temp),
                reduction="batchmean"
            ) * (temp ** 2)

            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item() if loss.item() > 0 else -loss.item()

            all_preds.extend(torch.sigmoid(student_outputs.logits).detach().cpu().numpy())
            all_truths.extend(batch["labels"].detach().cpu().numpy())

        avg_loss = total_loss / len(train_dl)
        compute_and_log_metrics(epoch+1, "train", np.array(all_preds), np.array(all_truths), avg_loss)

        student.eval()
        test_loss, test_preds, test_truths = 0, [], []
        with torch.no_grad():
            for batch in test_dl:
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = student(**batch)
                test_loss += F.binary_cross_entropy_with_logits(outputs.logits, batch["labels"]).item()
                test_preds.extend(torch.sigmoid(outputs.logits).cpu().numpy())
                test_truths.extend(batch["labels"].cpu().numpy())

        avg_test_loss = test_loss / len(test_dl)
        test_preds_np = np.array(test_preds)
        test_truths_np = np.array(test_truths)

        loss, micro, macro, acc, per_class_f1 = compute_and_log_metrics(
            epoch + 1, f"test", test_preds_np, test_truths_np, avg_test_loss, return_scores=True
        )

        if acc > best_test_acc:
            best_test_acc = acc
            patience_counter = 0
            print(f"New best test accuracy: {acc:.4f}")
            student.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
    start = time.time()
    with torch.no_grad():
        for batch in test_dl:
            _ = student(**{k: v.cuda() for k, v in batch.items()})
    end = time.time()
    latency_ms = (end - start) / len(test_dl.dataset) * 1000
    print(f"Student latency: {latency_ms:.3f} ms")
    teacher_param_count, _ = get_model_stats(teacher, name="Teacher")
    student_param_count, _ = get_model_stats(student, name="Student")
    teacher_latency, *_ = evaluate_model(teacher, test_dl)
    student_latency, *_ = evaluate_model(student, test_dl)

    compression_ratio = teacher_param_count / student_param_count
    speedup = teacher_latency / student_latency
    results_report = {
        "alpha": alpha,
        "temperature": temp,
        "loss": loss,
        "micro_f1": micro,
        "macro_f1": macro,
        "accuracy": acc,
    }
    for i, score in enumerate(per_class_f1):
        results_report[f"F1_class_{i}"] = score

    report_path = "./student_metrics/student_distill_hparam_results.csv"
    write_header = not os.path.exists(report_path)
    with open(report_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_report.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(results_report)

    results.append((alpha, temp, loss, acc))

results.sort(key=lambda x: x[3], reverse=True) 
best_alpha, best_temp, best_loss, best_acc = results[0]
print(f"\nBest α={best_alpha}, T={best_temp} | Test Accuracy: {best_acc:.4f}")

with open("./student_metrics/student_best_distill_config.txt", "w") as f:
    f.write(f"alpha={best_alpha}\ntemperature={best_temp}\ntest_accuracy={best_acc:.4f}")

df = pd.read_csv(METRIC_FILE)

plot_metric(df,"loss", "Loss over Epochs", "Loss", "student_loss.png")
plot_metric(df,"micro_f1", "Micro F1 over Epochs", "Micro F1", "student_micro_f1.png")
plot_metric(df,"macro_f1", "Macro F1 over Epochs", "Macro F1", "student_macro_f1.png")
plot_metric(df,"accuracy", "Accuracy over Epochs", "Accuracy", "student_accuracy.png")

results = []

print ("Compute Compression/Speedup ...")
for model_name, model_path in [("Teacher", "teacher_model"), ("Student", "student_model")]:
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda").eval()
    params, size_mb = get_model_stats(model, name=model_name)
    latency, acc, micro, macro = evaluate_model(model, test_dl)
    results.append({
        "Model": model_name,
        "Params": params,
        "Size_MB": size_mb,
        "Latency_ms": latency,
        "Accuracy": acc,
        "Micro_F1": micro,
        "Macro_F1": macro,
    })
    del model; torch.cuda.empty_cache()

base = results[0]
for r in results:
    r["Compression_Ratio"] = base["Params"] / r["Params"]
    r["Speedup"] = base["Latency_ms"] / r["Latency_ms"]

df = pd.DataFrame(results)
df.to_csv(f"{SAVE_DIR}/teacher_student_comparison.csv", index=False)
print(df)
