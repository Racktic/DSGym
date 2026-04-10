# MLE-Bench Integration Research

Date: 2026-04-08
Goal: Integrate MLE-Bench (full minus lite) into DSGym as a new `dspredict` split.

## 1. About MLE-Bench

MLE-Bench is OpenAI's benchmark of real Kaggle competitions for evaluating ML
agents (paper: "MLE-bench: Evaluating Machine Learning Agents on Machine
Learning Engineering", Chan et al., OpenAI 2024). The official repo is
https://github.com/openai/mle-bench and is already vendored locally at:

`/data/fnie/qixin/DSGym/examples/MLE_Bench_Eval/mle-bench/`

The repo defines several "splits" under `experiments/splits/`:

| split file | count | meaning |
|---|---|---|
| `low.txt` | 22 | **MLE-Bench Lite** (low complexity) |
| `medium.txt` | 38 | medium complexity |
| `high.txt` | 14 | high complexity |
| `split75.txt` | 74 | the canonical "full" set used in the paper (1 competition, `paddy-disease-classification` (=spaceship-titanic placeholder), is dev-only and not in split75; the README states "75 competitions", split75 contains 74 — see note below) |
| `dev.txt` | 6 | dev set |
| `systemcard.txt` | 29 | system-card subset |

Note: The README says 75 competitions; the actual `split75.txt` file contains
74 entries (one of the original 75, `paddy-disease-classification`, was dropped
or moved to dev). For our purposes, we use `split75.txt` as the authoritative
"full" set.

Each competition lives under `mlebench/competitions/<name>/` with
`config.yaml`, `description.md`, `prepare.py`, `grade.py`, `checksums.yaml`,
`leaderboard.csv`, etc.

## 2. Full set vs Lite

- **Full** (`split75.txt`): 74 competitions
- **Lite** (`low.txt`): 22 competitions
- **Full minus Lite**: **52 competitions**

(53 was the rough estimate in the task; the exact number is 52.)

## 3. Overlap with existing DSGym splits

Existing splits checked:
- `easy.json` — 38 tasks (Kaggle playground series)
- `hard.json` — 54 tasks (real Kaggle competitions)
- `mle_dojo.json` — 60 tasks (MLE-Dojo benchmark)

### Overlap of MLE-Bench full (74) with all DSGym splits

Only **6 overlaps**, all in `hard.json`:

1. `champs-scalar-coupling`
2. `stanford-covid-vaccine`
3. `statoil-iceberg-classifier-challenge`
4. `tensorflow-speech-recognition-challenge`
5. `tgs-salt-identification-challenge`
6. `ventilator-pressure-prediction`

`easy.json` and `mle_dojo.json` have **zero** overlap with MLE-Bench.

### Overlap of MLE-Bench (full minus lite, 52) with existing DSGym

Same 6 — none of these 6 are in `low.txt`, so they all sit in the "non-lite"
portion. After removing them, **the new split would contain 46 tasks**.

(If we instead define the new split as "full minus lite minus existing-DSGym",
the count is 52 - 6 = **46**.)

## 4. Recommended new split: `mle_bench` (46 tasks)

These are MLE-Bench full ∖ MLE-Bench-lite ∖ DSGym existing splits:

```
3d-object-detection-for-autonomous-vehicles
AI4Code
alaska2-image-steganalysis
billion-word-imputation
bms-molecular-translation
cassava-leaf-disease-classification
cdiscount-image-classification-challenge
chaii-hindi-and-tamil-question-answering
facebook-recruiting-iii-keyword-extraction
freesound-audio-tagging-2019
google-quest-challenge
google-research-identify-contrails-reduce-global-warming
h-and-m-personalized-fashion-recommendations
herbarium-2020-fgvc7
herbarium-2021-fgvc8
herbarium-2022-fgvc9
hms-harmful-brain-activity-classification
hotel-id-2021-fgvc8
hubmap-kidney-segmentation
icecube-neutrinos-in-deep-ice
imet-2020-fgvc7
inaturalist-2019-fgvc6
iwildcam-2019-fgvc6
iwildcam-2020-fgvc7
jigsaw-unintended-bias-in-toxicity-classification
kuzushiji-recognition
learning-agency-lab-automated-essay-scoring-2
lmsys-chatbot-arena
multi-modal-gesture-recognition
nfl-player-contact-detection
osic-pulmonary-fibrosis-progression
petfinder-pawpularity-score
plant-pathology-2021-fgvc8
predict-volcanic-eruptions-ingv-oe
rsna-2022-cervical-spine-fracture-detection
rsna-breast-cancer-detection
rsna-miccai-brain-tumor-radiogenomic-classification
seti-breakthrough-listen
siim-covid19-detection
smartphone-decimeter-2022
tensorflow2-question-answering
tweet-sentiment-extraction
us-patent-phrase-to-phrase-matching
uw-madison-gi-tract-image-segmentation
vesuvius-challenge-ink-detection
vinbigdata-chest-xray-abnormalities-detection
whale-categorization-playground
```

Notes about characteristics:
- Heavily skewed toward **large vision / multi-modal / segmentation /
  medical imaging** competitions (herbarium, iwildcam, hubmap, rsna, siim,
  vesuvius, vinbigdata, etc.). Disk footprint is huge — many of these
  datasets are tens to hundreds of GB.
- Several NLP tasks (lmsys-chatbot-arena, us-patent, learning-agency-lab,
  google-quest, jigsaw-unintended-bias, tweet-sentiment, billion-word,
  facebook-recruiting-iii, chaii, tensorflow2-question-answering).
- Audio (freesound, multi-modal-gesture).
- A few tabular/regression (predict-volcanic-eruptions, icecube-neutrinos).

## 5. Data download mechanism

### MLE-Bench's approach

MLE-Bench downloads via the **official Kaggle API** (`kaggle competitions
download -c <id>`) wrapped in `mlebench/data.py::download_dataset`. Then for
each competition there is a custom `prepare.py` that:

1. Extracts the raw zip into `raw/`.
2. Splits the original Kaggle public train into a new train + held-out test
   (because the original Kaggle private test labels are not available).
3. Writes `prepared/public/` (what the agent sees) and `prepared/private/`
   (held-out answers + grader inputs).
4. Verifies SHA-256 checksums recorded in `checksums.yaml`.

Key entry point: `mlebench/data.py::download_and_prepare_dataset(competition)`
and CLI `mlebench prepare -c <id>` (see `mlebench/cli.py`).

### Can we reuse `KaggleChallengeDownloader`?

`/data/fnie/qixin/DSGym/dsgym/datasets/loaders/kaggle_downloader.py` uses the
new `kagglesdk` python client and downloads the raw competition zip and scrapes
the description page. It can fetch the raw data, **but it does NOT do the
public/private split** that MLE-Bench needs to grade properly without leaking
the Kaggle test labels.

**Recommendation:** do NOT reinvent. For each MLE-Bench competition we should
run MLE-Bench's own `mlebench prepare -c <id>` (which already has per-task
`prepare.py` scripts), then copy/symlink the produced `prepared/public/` tree
into `data/data/dspredict-mle_bench/<competition_name>/` so it matches the
DSGym layout used by other dspredict splits. The held-out
`prepared/private/test.csv` (the answer key) and `grade.py` should also be
preserved so we can grade locally.

Concretely the per-competition layout should look like:

```
data/data/dspredict-mle_bench/<name>/
    train.csv / train/...        (from prepared/public)
    test.csv  / test/...
    sample_submission.csv
    description.md               (from mlebench/competitions/<name>/description.md)
    _private/
        test.csv                 (answer key, never mounted into agent container)
        grade.py
```

### Practical concerns
- **Disk:** estimate 1-3 TB total for 46 competitions. Need to pick a large
  storage volume; not all competitions are needed at once.
- **Kaggle TOS:** each competition requires accepting the rules on the website
  via the logged-in Kaggle account before the API will allow download. The
  existing DSPredict-Hard data already required this for many competitions.
- **Checksums:** MLE-Bench's `checksums.yaml` may be stale for re-released
  data; we may have to skip verification with `--skip-verification` for some.

## 6. Task JSON schema for the new split

Each entry in `data/task/dspredict/mle_bench.json` should follow the same
schema as `hard.json`:

```json
{
  "challenge_name": "<kaggle-id>",
  "description": "<contents of mlebench/competitions/<name>/description.md prefixed with 'Challenge description:\\n'>",
  "competition_description": "<short blurb / first paragraph>",
  "docker_challenge_path": "/data/<challenge_name>",
  "evaluation_metric": "<from config.yaml grader.name>",
  ...
}
```

The description, evaluation metric and dataset entrypoints can all be parsed
directly out of MLE-Bench's per-competition `config.yaml` + `description.md`,
so generation can be fully automated.

## 7. Suggested integration steps

1. **Generate `mle_bench.json`** with a small script that walks
   `examples/MLE_Bench_Eval/mle-bench/mlebench/competitions/<name>/` for the
   46 names listed above and emits hard-style task entries. Reuse the
   `description.md` file directly for the `description` field.
2. **Register the split** in
   `dsgym/datasets/loaders/dspredict.py::SPLIT_CONFIG`:
   ```python
   "mle_bench": ("mle_bench.json", "dspredict-mle_bench"),
   ```
3. **Provision data** with a helper script
   (`scripts/prepare_mle_bench_data.sh`) that loops over the 46 ids and runs:
   ```bash
   mlebench prepare -c <id> --data-dir /path/to/mlebench_cache
   ```
   then `rsync`/symlinks `prepared/public/` into
   `data/data/dspredict-mle_bench/<id>/`. Keep `prepared/private/` outside the
   agent's mount.
4. **Grader integration** — DSGym already has its own grading path; either:
   (a) reuse MLE-Bench's per-task `grade.py` by importing it, or
   (b) extract just the evaluation metric and write a thin adapter using
   DSGym's existing grading infra. (a) is faster to ship.
5. **Sanity-run** a couple of small ones first
   (`random-acts-of-pizza`-style — but those are all in lite already, so pick
   the smallest non-lite ones: `predict-volcanic-eruptions-ingv-oe`,
   `us-patent-phrase-to-phrase-matching`, `tweet-sentiment-extraction`).
6. **Decide whether to also onboard the 6 overlapping tasks** with MLE-Bench's
   prepared/private split, since the current `hard.json` versions probably use
   a different grader and a different held-out test set. Recommendation: leave
   them in `hard` and do not duplicate.

## 8. Open questions

- Do we want **only the 46 (non-lite, non-overlap)** or the full **52
  (non-lite)** including the 6 overlaps under a separate name? The cleanest
  story for benchmarking is "MLE-Bench non-lite = 52" so that comparisons to
  the paper map 1:1; recommend producing both `mle_bench` (52, full non-lite)
  and a derived view that excludes the 6 overlaps when reporting.
- Storage location and quota for ~1-3 TB of raw Kaggle data.
- Whether to mirror MLE-Bench's `prepared/public` exactly (which slightly
  alters the train/test split vs. the original Kaggle competition) or to use
  the original Kaggle splits with our own held-out scheme. MLE-Bench's
  approach is recommended for paper-compatibility.
