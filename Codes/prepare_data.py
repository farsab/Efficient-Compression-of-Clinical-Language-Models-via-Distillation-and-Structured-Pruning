import os
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

TOP_K = 50
BASE_DIR = r".\MIMIC -III (10000 patients)"
NOTE_PATH = os.path.join(BASE_DIR, "NOTEEVENTS", "NOTEEVENTS_sorted.csv")
DIAG_PATH = os.path.join(BASE_DIR, "DIAGNOSES_ICD", "DIAGNOSES_ICD_sorted.csv")
SAVE_DIR = "processed_mimic10k"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading NOTEEVENTS...")
notes = pd.read_csv(NOTE_PATH, usecols=["HADM_ID", "CATEGORY", "TEXT"])

print("Loading DIAGNOSES_ICD...")
diag = pd.read_csv(DIAG_PATH, usecols=["HADM_ID", "ICD9_CODE"])

notes = notes[notes["CATEGORY"] == "Discharge summary"]
notes = notes.dropna(subset=["TEXT", "HADM_ID"])

print("Joining notes with diagnoses...")
merged = notes.merge(diag, on="HADM_ID")
grouped = merged.groupby("HADM_ID").agg({
    "TEXT": "first",  # one note per admission
    "ICD9_CODE": lambda x: list(x)
}).reset_index()

print(f"Keeping top {TOP_K} ICD codes...")
all_codes = [c for sublist in grouped["ICD9_CODE"] for c in sublist]
top_codes = pd.Series(all_codes).value_counts().nlargest(TOP_K).index.tolist()

grouped["labels"] = grouped["ICD9_CODE"].apply(lambda codes: [c for c in codes if c in top_codes])
grouped = grouped[grouped["labels"].map(len) > 0]

mlb = MultiLabelBinarizer(classes=top_codes)
y = mlb.fit_transform(grouped["labels"])
X = grouped["TEXT"].tolist()

print("Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def save(obj, name):
    with open(os.path.join(SAVE_DIR, name), "wb") as f:
        pickle.dump(obj, f)

print("Saving processed files...")
save(X_train, "X_train.pkl")
save(X_test, "X_test.pkl")
save(y_train, "y_train.pkl")
save(y_test, "y_test.pkl")
save(mlb, "label_binarizer.pkl")

print("Done. Files saved to:", SAVE_DIR)
