# 🩺 Diabetes Progression Predictor

> *"Which patient measurements actually predict disease progression? Let 442 real patient records answer instead of guessing."*

A beginner regression project on the **Diabetes** dataset — 442 real patients, 10 baseline clinical measurements, one year of progression as the target. It was my first time building something with medical data, and it changed how careful I became about what a model can and cannot tell you.

## What this project does

- Loads the real Diabetes dataset (442 patients, 10 features).
- Compares Linear Regression, Random Forest, and Gradient Boosting.
- Reports MAE, RMSE, and R² on a held-out test split.
- Ranks features by importance and by Pearson correlation.
- Includes a CLI to estimate progression for a given patient.

## The dataset

[Diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) — 442 patients, one year after baseline. Ten features are **standardised**, so a value like `0.05` for `bmi` means "slightly above average".

| Feature | Notes |
|---|---|
| `age`, `sex` | Standardised |
| `bmi` | Body mass index, standardised |
| `bp` | Blood pressure, standardised |
| `s1`–`s6` | Six blood serum measurements, standardised |
| `disease_progression` | One-year target — **target** |

## How to run it

```bash
pip install -r requirements.txt

python train.py        # load data, train, compare models
python -m src.explore  # correlation analysis + plots

# Estimate progression for one patient
python -m src.predict --age 0.03 --sex 0.05 --bmi 0.06 --bp 0.02 \
    --s1 -0.01 --s2 0.06 --s3 -0.01 --s4 -0.02 --s5 0.02 --s6 -0.02
```

## Project structure

```
diabetes-progression-predictor/
├── data/
│   └── diabetes.csv
├── src/
│   ├── load_data.py      # fetch + cache
│   ├── explore.py        # correlation analysis
│   ├── train_model.py    # train, compare, evaluate
│   └── predict.py        # CLI
├── tests/
├── train.py
├── requirements.txt
└── README.md
```

## What I learned

- That BMI is the strongest single predictor here — and that matches clinical intuition.
- How to read feature importance without overclaiming causation.
- Why R² around 0.45 on medical data is respectable, not disappointing.
- That a model predicting a patient's health carries responsibility, and this project pushed me to be explicit about that in every output.

## Results

On a 20% held-out test set:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| LinearRegression | 42.79 | 53.85 | 0.453 |
| RandomForest | 44.58 | 54.76 | 0.434 |
| **GradientBoosting** | **44.53** | **53.73** | **0.455** |

R² around 0.45 is close to what the literature reports for this dataset. BMI (0.39) and `s5` (0.25) dominate feature importance.

---

*Built with Python, pandas, scikit-learn, matplotlib, seaborn. Real clinical data, honestly measured.*
