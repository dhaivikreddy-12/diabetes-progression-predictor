# 📚 Student Marks Predictor

> *"How much does study time actually matter? Let the data answer instead of guessing."*

A beginner regression project that predicts a student's final exam score from simple daily habits like study hours, sleep, and attendance. It also digs into the data to show which factors matter most — because in real life, data tells better stories than opinions.

## What this project does

- Loads a realistic dataset of student records.
- Does EDA with charts (study hours vs marks, attendance vs marks).
- Trains a **Random Forest Regressor** to predict exam scores.
- Reports feature importance — so you can see what actually drives performance.
- Predicts a score for any student you describe.

## The dataset

Synthetic but realistic (`data/student_marks.csv`, ~300 students):

| Feature             | Description                          |
|---------------------|--------------------------------------|
| `study_hours`       | Avg study hours per day              |
| `sleep_hours`       | Avg sleep hours per night            |
| `attendance_pct`    | Class attendance percentage          |
| `previous_score`    | Score in the previous exam (%)       |
| `final_score`       | Final exam score (%) — target        |

## How to run it

```bash
pip install -r requirements.txt

# Generate data, train, and evaluate
python train.py

# Explore the data visually
python explore.py   # saves plots to plots/

# Predict a score
python predict.py --study 6 --sleep 7 --attendance 90 --prev 78
```

## What I learned

- That correlation doesn't equal causation, but it's a great starting point.
- How Random Forests combine many small trees into one stronger model.
- How to read feature importance to tell a real story from the data.
- How to think about "good enough" accuracy for a real-world-ish problem.

## Results

The model reaches an **R² around 0.90** on held-out students, with `study_hours` and `attendance_pct` coming out as the strongest predictors — a nice, intuitive finding that the numbers back up.

---

*Built with Python, pandas, scikit-learn, matplotlib. Made for learning, by a student, for students.*
