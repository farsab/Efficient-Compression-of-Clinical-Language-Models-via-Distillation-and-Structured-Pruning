@echo off
python train_teacher.py
python prune_teacher.py
python train_student.py
python prune_student.py
