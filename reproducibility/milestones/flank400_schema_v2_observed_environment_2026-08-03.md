# Flank-400 schema-v2 observed environment

Capture date: 2026-08-03.

Repository commit: `fac6ded9c8c3a3e1d1eb2f77f512d4f5a7c0121c`.

Status: observed post-milestone project environment.

This is an exact record of the environment observed on the capture date. It is not claimed to be a historical snapshot of the environment originally used to train either selected checkpoint.

CUDA was unavailable to PyTorch during capture. Therefore, this record does not independently validate the original GPU training or GPU inference runtime.

## System and runtime

```text
Linux DESKTOP-2RJ74O3 4.4.0-19041-Microsoft #5794-Microsoft Mon Apr 07 17:55:00 PST 2025 x86_64 x86_64 x86_64 GNU/Linux
PRETTY_NAME="Ubuntu 24.04.1 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.1 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=noble
LOGO=ubuntu-logo
Python 3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]
PYTHON_EXECUTABLE=/home/unix_s/projects/calibrated-splice-prediction/.venv_wsl/bin/python
PYTHON_PREFIX=/home/unix_s/projects/calibrated-splice-prediction/.venv_wsl
OPEN_SPLICEAI_VERSION=0.0.5
OPEN_SPLICEAI_LOCATION=/home/unix_s/projects/calibrated-splice-prediction/.venv_wsl/lib/python3.12/site-packages
TORCH_VERSION=2.12.0+cu130
TORCH_CUDA_VERSION=13.0
CUDNN_VERSION=92000
CUDA_AVAILABLE=False
```

## Exact observed package set

Dependency lock: `reproducibility/milestones/flank400_schema_v2_observed_requirements_2026-08-03.txt`.

Dependency-lock SHA-256: `162934480bc33efbac2e081518f2a2bf313c5a8d9155019b7163805484f6aa44`.
