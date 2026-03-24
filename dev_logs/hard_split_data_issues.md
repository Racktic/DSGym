# DSPredict Hard Split 数据问题报告

## 背景

使用 DSGym 自带的 `KaggleChallengeDownloader`（kagglesdk + zipfile.extractall）下载全部 54 个 hard split 竞赛数据后，发现部分任务的实际文件结构与 `hard.json` 中 `description` 字段描述的不一致，导致 agent 按 description 中的路径访问文件时报 FileNotFoundError。

根因：Kaggle API 下载的 zip 包内部目录结构因竞赛而异，`extractall` 保留了 zip 内部的子目录结构，但 description（来自 Kaggle 竞赛页面）中引用的是 Kaggle notebook 环境下的扁平路径。

## 已修复的问题

### 类型 1：文件在子目录中，description 用裸文件名（4 个）

Description 中写 `Teams.csv`，但实际文件在 `DataFiles/Teams.csv`。

| 竞赛 | 实际子目录 | 修复方式 |
|---|---|---|
| mens-machine-learning-competition-2018 | `DataFiles/` | `ln -s DataFiles/* .` |
| mens-machine-learning-competition-2019 | `DataFiles/` | `ln -s DataFiles/* .` |
| mens-march-mania-2022 | `MDataFiles_Stage1/`, `MDataFiles_Stage2/` | `ln -s MDataFiles_Stage*/* .` |
| womens-machine-learning-competition-2019 | `WDataFiles/`, `Stage2WDataFiles/` | `ln -s WDataFiles/* . && ln -s Stage2WDataFiles/* .` |

### 类型 2：目录双层嵌套（1 个）

| 竞赛 | 问题 | 修复方式 |
|---|---|---|
| trec-covid-information-retrieval | `CORD-19/CORD-19/metadata.csv`，description 期望 `CORD-19/metadata.csv` | `ln -s CORD-19/CORD-19/* CORD-19/` |

### 类型 3：Description 期望的目录不存在（1 个）

| 竞赛 | 问题 | 修复方式 |
|---|---|---|
| geolifeclef-2024 | Description 提到 `PresenceOnlyOccurences/GLC24_PO_metadata_train.csv` 和 `PresenceAbsenceSurveys/GLC24_PA_metadata_train.csv`，实际 CSV 在根目录且文件名不同（`GLC24_P0_metadata_train.csv`）。EnvironmentalRasters 也有双层嵌套（`EnvironmentalRasters/EnvironmentalRasters/Climate/`） | `mkdir PresenceOnlyOccurences PresenceAbsenceSurveys` + `cp` 文件进去；EnvironmentalRasters 下建软链消除双层嵌套 |

### 类型 4：Kaggle 下载的文件名与 description 不一致（6 个）

这是 Kaggle 数据本身的问题，与下载方式无关。

| 竞赛 | Description 写的 | 实际文件名 | 修复方式 |
|---|---|---|---|
| ashrae-energy-prediction | `building_meta.csv` | `building_metadata.csv` | `cp` 一份 |
| rsna-pneumonia-detection-challenge | `stage_2_train.csv` | `stage_2_train_labels.csv` | `cp` 一份 |
| planttraits2024 | `target_name_meta.csv` | `target_name_meta.tsv` | `cp` 改扩展名 |
| imaterialist-challenge-fashion-2018 | `sample_submission_randomlabel.csv` | `sample_submission.csv` | `cp` 一份 |
| talkingdata-adtracking-fraud-detection | `sampleSubmission.csv` | `sample_submission.csv` | `cp` 一份 |
| inclusive-images-challenge | `stage_2_test_images/` | `stage_2_images/` | `ln -s` |

### 类型 5：下载后文件仍为压缩格式（6 个）

KaggleChallengeDownloader 解压了外层 zip，但部分竞赛的数据文件本身又是 zip/7z 格式。

| 竞赛 | 格式 | 修复方式 |
|---|---|---|
| imaterialist-challenge-fashion-2018 | 内层 `.zip` | `unzip *.zip` |
| recruit-restaurant-visitor-forecasting | 内层 `.zip` | `unzip *.zip` |
| web-traffic-time-series-forecasting | 内层 `.zip` | `unzip *.zip` |
| data-science-bowl-2018 | 内层 `.zip`（图像） | `unzip *.zip` |
| statoil-iceberg-classifier-challenge | `.7z` 格式 | `7z x *.7z` + `ln -s` |
| tensorflow-speech-recognition-challenge | `.7z` 格式（3.5GB） | **未解压**（见下方） |

### 类型 6：Kaggle 竞赛描述中的拼写错误（1 个）

| 竞赛 | 问题 | 修复方式 |
|---|---|---|
| LANL-Earthquake-Prediction | Kaggle 原始 description 写 `sample_sumbission.csv`（拼写错误），实际文件名 `sample_submission.csv` | 手动改文件名 |

## 未修复的问题

### 1. tensorflow-speech-recognition-challenge

数据为 `.7z` 格式（train.7z 约 1GB，test.7z 约 2.5GB），解压耗时过长且磁盘空间紧张，暂未解压。该任务还需要 `librosa` 等音频处理库，当前 `executor-kaggle` 容器不支持。

### 2. 容器环境缺失（需要 mle_image）

以下竞赛需要 CV/NLP/音频相关的 Python 包，`executor-kaggle` 容器中未安装：

| 缺失模块 | 影响的竞赛 |
|---|---|
| tensorflow | data-science-bowl-2018, humpback-whale-identification, rsna-pneumonia-detection-challenge, sp-society-camera-model-identification, siim-acr-pneumothorax-segmentation |
| cv2 (opencv) | 同上多个 CV 竞赛 |
| skimage (scikit-image) | data-science-bowl-2018, sp-society-camera-model-identification |
| pydicom | rsna-pneumonia-detection-challenge, siim-acr-pneumothorax-segmentation |
| librosa | tensorflow-speech-recognition-challenge |
| torch_geometric | predict-ai-model-runtime |
| rank_bm25 | trec-covid-information-retrieval |

这些需要构建 `executor-mle` 镜像（基于 mle_image Dockerfile，包含 93 个 ML 包）后才能支持。

## 建议

1. **KaggleChallengeDownloader 改进**：下载后递归解压内层 zip/7z 文件，并将子目录中的文件 flatten 到竞赛根目录。
2. **hard.json 后处理脚本**：提供一个验证脚本，检查每个 task 的 description 中提到的文件是否在对应数据目录中存在。
3. **容器镜像选择**：对于 CV/NLP/音频任务，应使用 `executor-mle`（或类似的全栈镜像）而非 `executor-kaggle`。
