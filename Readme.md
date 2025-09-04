#  Efficient Compression of Clinical Language Models via Distillation and Structured Pruning 

This repository provides a full pipeline to **train a large clinical language model**, apply **structured pruning**, and perform **knowledge distillation** to a compact student model. The goal is to improve computational efficiency **without significantly sacrificing performance**.

---

##  Project Structure

| Script                      | Description                                      |
|----------------------------|--------------------------------------------------|
| `train_teacher.py`         | Fine-tunes the large teacher model               |
| `prune_teacher.py`         | Applies structured pruning to the teacher        |
| `train_student.py`         | Trains a compact student using distillation      |
| `prune_student.py`         | Applies pruning to the student model             |
| `prepare_data.py`          | (Optional) Processes raw MIMIC-III into inputs   |
| `*.bat` files              | Batch scripts for full pipeline execution        |

---

## Environment Setup

Set up your environment using Anaconda:

```bash
conda create -n hsj_env python=3.10.18 -y
conda activate hsj_env
pip install -r requirements.txt
```

---

##  Dataset

This pipeline uses the **MIMIC-III dataset** (10,000 patients sample).

- Place the preprocessed dataset folder `processed_mimic10k/` in the root of the repo.
- If you'd rather process raw data yourself:
  ```bash
  python prepare_data.py
  ```
   _Note: The processed version is already included. Running `prepare_data.py` is optional unless starting from scratch._

---

##  Running the Pipeline


# Running the Pipeline

You can execute the full training–pruning–distillation pipeline in **two ways**:

## Option 1: Configurable Execution via `run.bat`

Use the `run.bat` script to activate the environment and run the desired pipeline:

```bash
run.bat [conda_env] [A|B|C]
```

- `conda_env`: Name of your Conda environment (e.g., `hsj_env`)
- `A`, `B`, or `C`: Pipeline variant to run (see below)

> **Important:** Always run this inside an **Anaconda Prompt**, not a regular CMD window.

---

## Option 2: Manual Execution of Batch Files

Run each experiment individually using the corresponding batch file:

### 🔹 A. Knowledge Distillation Only: Train Teacher → Train Student
```bash
run_teacher_then_student.bat
```

### 🔹 B. Train → Distill → Prune Student Only
```bash
run_teacher_student_then_prune_student.bat
```

### 🔹 C. Full Pipeline: Train → Prune Teacher → Train → Prune Student
```bash
run_all_teacher_to_pruned_student.bat
```

---

##  Outputs

All output files and logs are saved automatically. Several additional figures are also generated to support further verification, though they are not discussed in the report due to space limitations.

| Folder                      | Contents                                                  |
|----------------------------|-----------------------------------------------------------|
| `teacher_model/`           | Trained teacher model checkpoints                         |
| `pruned_teacher_model/`    | Pruned teacher checkpoints                                |
| `student_model/`           | Distilled student model                                   |
| `pruned_student_model/`    | Pruned student model                                      |
| `teacher_metrics/`         | Accuracy, F1, latency, and size logs for teacher          |
| `student_metrics/`         | Accuracy, F1, latency, and size logs for student          |

All metrics include:

- **Accuracy**
- **Micro/Macro F1**
- **Latency (ms)**
- **Model size (MB)**
- **Parameter count**
- **Compression ratio**
- **Speedup factor**

---

## Tips & Notes

- Batch size, epochs, patience, temperature, and alpha can be configured inside each script.
- Make sure your environment has GPU support for faster training.
- If encountering `UnicodeEncodeError` on Windows, open the CSVs using UTF-8 encoding or modify writer line in the script.
- All the code files are reused across different experimental cases, which means the generated models and metric files will be overwritten if run in the same directory. 
  To avoid this, the recommended practice is to create a separate folder for each case, and copy all the relevant files—including the batch file—into it. This will preserve the outputs for each case without conflicts.

---

## Questions or Issues?

If anything is unclear, or if you run into issues, feel free to open an issue in the repository. We've tried to make the process smooth — your feedback is appreciated.
