# Student Performance Prediction System

Classroom prototype that flags students who may be **at risk** of failing the final Mathematics grade, built with Python, scikit-learn and Streamlit.

## Problem
Academic risk is often noticed only after poor results. This prototype explores early identification using available student indicators (study time, absences, failures, G1, G2, and others).

## Dataset
UCI Machine Learning Repository, *Student Performance* (ID 320):
https://archive.ics.uci.edu/dataset/320/student+performance

## Prediction target
`G3 < 10` = **AT RISK**; `G3 >= 10` = **NOT AT RISK**.
`G3` is the target only and is never used as an input feature (avoids target leakage).

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest), joblib, Streamlit, GitHub.

## Features
- **Prediction tab:** dataset summary, sample CSV download, CSV upload with required-column check, predictions, and downloadable results.
- **Model evaluation tab:** accuracy, precision, recall, F1-score and confusion matrix from the last training run.
- **Midterm Examination tab:** 20 multiple-choice items (auto-graded with explanations) and 4 short-answer questions, with a downloadable exam record.

## Installation and run
```bash
pip install -r requirements.txt
python train_model.py      # creates student_data.csv, student_model.pkl, metrics.json
streamlit run app.py       # opens http://localhost:8501
```

## Project structure
```
student-performance-prediction/
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

## Limitations
- Trained on one Portuguese secondary-school dataset; it may not generalize to other institutions or populations.
- Predictions are not diagnoses and must not be the sole basis for real student decisions.
- Never commit passwords or access tokens to the repository.
