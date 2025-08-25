@echo off
if "%~1"=="" (
    echo [ERROR] Missing conda environment name.
    echo Usage: run.bat [conda_env] [A/B/C]
    exit /b 1
)
if "%~2"=="" (
    echo [ERROR] Missing option A, B, or C.
    echo Usage: run.bat [conda_env] [A/B/C]
    exit /b 1
)

set ENV_NAME=%~1
set OPTION=%~2

call conda activate %ENV_NAME%

if /I "%OPTION%"=="A" (
    call run_teacher_then_student.bat
) else if /I "%OPTION%"=="B" (
    call run_teacher_student_then_prune_student.bat
) else if /I "%OPTION%"=="C" (
    call run_all_teacher_to_pruned_student.bat
) else (
    echo [ERROR] Invalid option: %OPTION%. Choose A, B, or C.
    exit /b 1
)

pause
