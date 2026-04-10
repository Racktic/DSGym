  Column 1: Dataset                                                                                       
  Standard (docker-compose.yml): DAEval, DiscoveryBench, DABStep, QRData                                  
  Bio (docker-compose-dsbio.yml): DSBio (bioinformatics)                                                  
  DSPredict (docker-dspredict-hard.yml): DSPredict (Kaggle competitions)                                  
  ────────────────────────────────────────                  
  Column 1: Image
  Standard (docker-compose.yml): executor-prebuilt
  Bio (docker-compose-dsbio.yml): executor-bio
  DSPredict (docker-dspredict-hard.yml): executor-kaggle
  ────────────────────────────────────────
  Column 1: Base
  Standard (docker-compose.yml): python:3.12-slim
  Bio (docker-compose-dsbio.yml): python:3.11-slim
  DSPredict (docker-dspredict-hard.yml): pytorch/pytorch:2.8.0-cuda12.6
  ────────────────────────────────────────
  Column 1: Containers
  Standard (docker-compose.yml): 64
  Bio (docker-compose-dsbio.yml): 64
  DSPredict (docker-dspredict-hard.yml): 8
  ────────────────────────────────────────
  Column 1: Memory/container
  Standard (docker-compose.yml): 2 GB
  Bio (docker-compose-dsbio.yml): 10 GB
  DSPredict (docker-dspredict-hard.yml): 24 GB
  ────────────────────────────────────────
  Column 1: CPUs/container
  Standard (docker-compose.yml): 0.5
  Bio (docker-compose-dsbio.yml): 0.5
  DSPredict (docker-dspredict-hard.yml): 8
  ────────────────────────────────────────
  Column 1: Timeout
  Standard (docker-compose.yml): 600s
  Bio (docker-compose-dsbio.yml): 1200s
  DSPredict (docker-dspredict-hard.yml): 3600s
  ────────────────────────────────────────
  Column 1: GPU
  Standard (docker-compose.yml): No
  Bio (docker-compose-dsbio.yml): No
  DSPredict (docker-dspredict-hard.yml): Yes (NVIDIA, 1 GPU per container)
  ────────────────────────────────────────
  Column 1: Key packages
  Standard (docker-compose.yml): numpy, pandas, scikit-learn, torch, causalml, dowhy, lingam
  Bio (docker-compose-dsbio.yml): scanpy, anndata, biopython, h5py, pybedtools, squidpy (biology-specific)
  DSPredict (docker-dspredict-hard.yml): xgboost, lightgbm, catboost, optuna (ML competition stack)
  ────────────────────────────────────────
  Column 1: Special feature
  Standard (docker-compose.yml): General DS tasks
  Bio (docker-compose-dsbio.yml): Bio/genomics libraries, writable submission dir
  DSPredict (docker-dspredict-hard.yml): Kaggle API submissions, GPU access

  Which to use:
  - Running DAEval / DiscoveryBench / DABStep / QRData → start docker-compose.yml
  - Running DSBio → start docker-compose-dsbio.yml
  - Running DSPredict (Kaggle competitions) → start docker-dspredict-hard.yml (requires GPU + Kaggle API
  key)