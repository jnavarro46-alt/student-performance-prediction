# app.py
import json
import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Predict", page_icon="🎓", layout="wide")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")


@st.cache_data
def load_metrics():
    if os.path.exists("metrics.json"):
        with open("metrics.json") as f:
            return json.load(f)
    return None


if not (os.path.exists("student_model.pkl") and os.path.exists("student_data.csv")):
    # First run (for example on Streamlit Cloud): download the data and train.
    import subprocess
    import sys

    with st.spinner("First run: downloading the dataset and training the model "
                    "(about 1 minute)..."):
        run = subprocess.run([sys.executable, "train_model.py"],
                             capture_output=True, text=True)
    if run.returncode != 0:
        st.error("Training failed. Make sure train_model.py is in the repository.")
        st.code((run.stdout + run.stderr)[-2000:])
        st.stop()
    st.cache_data.clear()
    st.cache_resource.clear()

model = load_model()
data = load_data()
metrics = load_metrics()

st.title("🎓 Student Performance Prediction")
st.caption("Classroom prototype for academic risk exploration. "
           "Predictions are not diagnoses and must not be the sole basis "
           "for real student decisions.")

tab_predict, tab_eval, tab_exam = st.tabs(
    ["📊 Prediction", "📈 Model evaluation", "📝 Midterm Examination"]
)

# ---------------------------------------------------------------------------
# TAB 1 - Prediction
# ---------------------------------------------------------------------------
with tab_predict:
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", len(data))
    c2.metric("Average final grade", round(data["G3"].mean(), 2))
    c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

    st.subheader("Dataset preview")
    st.dataframe(data.head(10))

    st.subheader("Predict uploaded student records")
    st.write("Upload a CSV containing the same predictor columns as the training data.")

    template = data.drop(columns=["G3"]).head(5)
    st.download_button(
        "Download sample input CSV",
        template.to_csv(index=False),
        "sample_prediction_input.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Choose student CSV", type=["csv"])

    if uploaded is not None:
        try:
            new_data = pd.read_csv(uploaded)
        except Exception:
            st.error("The file could not be read as a CSV. Please upload a valid CSV file.")
            new_data = None

        if new_data is not None:
            required_columns = list(model.feature_names_in_)
            missing = [c for c in required_columns if c not in new_data.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                predictions = model.predict(new_data[required_columns])
                results = new_data.copy()
                results["Prediction"] = [
                    "AT RISK" if value == 1 else "NOT AT RISK"
                    for value in predictions
                ]
                st.subheader("Prediction results")
                st.dataframe(results)
                st.metric("Predicted at-risk records", int((predictions == 1).sum()))
                st.download_button(
                    "Download prediction results",
                    results.to_csv(index=False),
                    "student_predictions.csv",
                    "text/csv",
                )

# ---------------------------------------------------------------------------
# TAB 2 - Model evaluation
# ---------------------------------------------------------------------------
with tab_eval:
    st.subheader("Model evaluation summary")
    if metrics is None:
        st.info("No metrics.json found. Run `python train_model.py` to generate it.")
    else:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.caption("Do not interpret accuracy alone. Check precision, recall and F1 "
                   "for the 'At risk' class.")
        rows = []
        for label in ["Not at risk", "At risk"]:
            r = metrics["report"][label]
            rows.append({
                "Class": label,
                "Precision": round(r["precision"], 3),
                "Recall": round(r["recall"], 3),
                "F1-score": round(r["f1-score"], 3),
                "Support": int(r["support"]),
            })
        st.dataframe(pd.DataFrame(rows))

        st.write("Confusion matrix (rows = actual, columns = predicted)")
        cm = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual: Not at risk", "Actual: At risk"],
            columns=["Predicted: Not at risk", "Predicted: At risk"],
        )
        st.dataframe(cm)
        st.caption(f"Training rows: {metrics['train_rows']} | "
                   f"Test rows: {metrics['test_rows']}")

# ---------------------------------------------------------------------------
# TAB 3 - Midterm examination
# ---------------------------------------------------------------------------
# Each item: question, options, index of the correct option, explanation.
EXAM = [
    {
        "q": "What is the prediction target in this laboratory?",
        "options": [
            "Whether a student's final Mathematics grade G3 is below 10 (at risk)",
            "The student's age",
            "The student's first-period grade G1",
            "The number of absences",
        ],
        "answer": 0,
        "why": "The system flags a student as AT RISK when G3 < 10.",
    },
    {
        "q": "Why is G3 excluded from the input features?",
        "options": [
            "It contains missing values",
            "It is categorical",
            "It would cause target leakage because the label is derived from it",
            "It is too large for the model",
        ],
        "answer": 2,
        "why": "The label is computed from G3, so using it as an input would let the model 'cheat'.",
    },
    {
        "q": "A student has G3 = 10. How is this student labelled?",
        "options": ["AT RISK", "NOT AT RISK", "Unknown", "Removed as a duplicate"],
        "answer": 1,
        "why": "The rule is G3 < 10 = AT RISK; G3 >= 10 = NOT AT RISK.",
    },
    {
        "q": "What does SimpleImputer(strategy='median') do to numeric columns?",
        "options": [
            "Deletes rows with missing values",
            "Replaces missing values with the column median",
            "Converts numbers to text",
            "Scales values to mean 0",
        ],
        "answer": 1,
        "why": "Imputation fills in missing values; the median is robust to outliers.",
    },
    {
        "q": "Why is StandardScaler applied to numeric variables?",
        "options": [
            "To put variables on a comparable scale (mean 0, standard deviation 1)",
            "To convert them to categories",
            "To remove the target column",
            "To balance the classes",
        ],
        "answer": 0,
        "why": "Scaling keeps variables with large ranges from dominating scale-sensitive models.",
    },
    {
        "q": "What does OneHotEncoder do?",
        "options": [
            "Fills missing values",
            "Splits data into train and test sets",
            "Turns each category of a text column into its own 0/1 column",
            "Saves the model to disk",
        ],
        "answer": 2,
        "why": "Models need numbers; one-hot encoding represents categories as binary columns.",
    },
    {
        "q": "What is the purpose of handle_unknown='ignore' in OneHotEncoder?",
        "options": [
            "It removes unknown columns from the CSV",
            "It prevents an error when a category not seen in training appears at prediction time",
            "It deletes rows with unknown grades",
            "It changes the target label",
        ],
        "answer": 1,
        "why": "Uploaded CSVs may contain categories that were not in the training data.",
    },
    {
        "q": "What does train_test_split(..., test_size=0.20, stratify=y) do?",
        "options": [
            "Uses 20% of the data for training",
            "Shuffles the columns",
            "Removes 20% of the at-risk students",
            "Holds out 20% for testing and keeps class proportions similar in both sets",
        ],
        "answer": 3,
        "why": "Stratifying keeps the at-risk ratio similar in the training and test sets.",
    },
    {
        "q": "What does class_weight='balanced' do in the Random Forest?",
        "options": [
            "Gives more weight to the minority class so it is not ignored",
            "Makes all trees identical",
            "Removes the majority class",
            "Increases the number of trees",
        ],
        "answer": 0,
        "why": "At-risk students are fewer, so the model is told to pay them more attention.",
    },
    {
        "q": "What does recall for the 'At risk' class tell you?",
        "options": [
            "Of the students predicted at risk, the share that truly were",
            "Of the students who were actually at risk, the share the model caught",
            "The share of all predictions that were correct",
            "The number of trees that agreed",
        ],
        "answer": 1,
        "why": "Recall = true positives / (true positives + false negatives). Low recall means missed students.",
    },
    {
        "q": "What does precision for the 'At risk' class tell you?",
        "options": [
            "Of the students actually at risk, the share the model caught",
            "The overall accuracy",
            "Of the students the model flagged as at risk, the share that truly were",
            "How fast the model runs",
        ],
        "answer": 2,
        "why": "Precision = true positives / (true positives + false positives).",
    },
    {
        "q": "Why should accuracy not be interpreted alone?",
        "options": [
            "Accuracy cannot be computed for classifiers",
            "Accuracy only works on training data",
            "Accuracy is always 100%",
            "It can look high even when the model misses many at-risk students",
        ],
        "answer": 3,
        "why": "With unequal class sizes, a model can be accurate overall yet weak on the class that matters.",
    },
    {
        "q": "Which library is used to save and load the trained model file student_model.pkl?",
        "options": ["joblib", "streamlit", "ucimlrepo", "matplotlib"],
        "answer": 0,
        "why": "joblib.dump saves the pipeline and joblib.load restores it in the app.",
    },
    {
        "q": "Which command starts the Streamlit app locally?",
        "options": [
            "python streamlit.py",
            "streamlit run app.py",
            "git run app.py",
            "pip run app.py",
        ],
        "answer": 1,
        "why": "streamlit run app.py serves the app, normally at http://localhost:8501.",
    },
    {
        "q": "Why is @st.cache_resource used on load_model()?",
        "options": [
            "To train the model again on every click",
            "To encrypt the model",
            "To load the model once and reuse it between app reruns",
            "To download the dataset",
        ],
        "answer": 2,
        "why": "Streamlit reruns the script on every interaction; caching avoids reloading the model each time.",
    },
    {
        "q": "Why does the app check required columns before predicting?",
        "options": [
            "To make the page load faster",
            "To change the target label",
            "To delete extra rows",
            "To show a helpful error instead of crashing when the CSV does not match the training data",
        ],
        "answer": 3,
        "why": "The model expects the same predictor columns it was trained on.",
    },
    {
        "q": "Which command uploads your local main branch to GitHub for the first time?",
        "options": [
            "git push -u origin main",
            "git init main",
            "git add origin",
            "git clone main",
        ],
        "answer": 0,
        "why": "It pushes main to the remote named origin and sets it as the upstream branch.",
    },
    {
        "q": "What should never be placed in your source code on GitHub?",
        "options": [
            "A README file",
            "Passwords or personal access tokens",
            "A requirements.txt file",
            "Comments explaining the code",
        ],
        "answer": 1,
        "why": "Anything pushed to a public repository can be read by anyone.",
    },
    {
        "q": "What is a key limitation of this model?",
        "options": [
            "It can only predict G3 exactly",
            "It runs only on Windows",
            "It was trained on one Portuguese school dataset, so it may not generalize to other institutions",
            "It cannot read CSV files",
        ],
        "answer": 2,
        "why": "Different schools and populations may have different patterns.",
    },
    {
        "q": "What is the most appropriate action after seeing an 'AT RISK' prediction?",
        "options": [
            "Fail the student immediately",
            "Ignore it, because models are always wrong",
            "Remove the student from the class",
            "Treat it as one signal, review other information, and offer academic support",
        ],
        "answer": 3,
        "why": "Predictions are not diagnoses and must not be the sole basis for decisions.",
    },
]

SHORT_ANSWER = [
    "What is the prediction target, and why is G3 excluded from the input features?",
    "What does the model's recall for the 'At risk' class tell you?",
    "Why might a model trained on this dataset perform differently on students from your institution?",
    "What action should a faculty member take after seeing an 'AT RISK' prediction?",
]

LETTERS = "ABCD"

with tab_exam:
    st.subheader("Midterm Examination")
    st.write(
        f"**Part I:** {len(EXAM)} multiple-choice items (1 point each, graded automatically).  \n"
        f"**Part II:** {len(SHORT_ANSWER)} short-answer questions (recorded in your download for your instructor to grade)."
    )

    result = st.session_state.get("exam_result")

    if result is None:
        with st.form("midterm_form"):
            name = st.text_input("Full name")
            section = st.text_input("Course / section")

            st.markdown("### Part I - Multiple choice")
            choices = []
            for i, item in enumerate(EXAM, start=1):
                choices.append(
                    st.radio(f"{i}. {item['q']}", item["options"],
                             index=None, key=f"mcq_{i}")
                )

            st.markdown("### Part II - Short answer")
            short = []
            for j, q in enumerate(SHORT_ANSWER, start=1):
                short.append(st.text_area(f"{j}. {q}", key=f"sa_{j}"))

            submitted = st.form_submit_button("Submit exam")

        if submitted:
            unanswered = [i + 1 for i, c in enumerate(choices) if c is None]
            if not name.strip():
                st.error("Please enter your full name before submitting.")
            elif unanswered:
                st.error("Please answer all items in Part I. Unanswered: "
                         + ", ".join(map(str, unanswered)))
            else:
                score = 0
                review = []
                for i, (item, chosen) in enumerate(zip(EXAM, choices), start=1):
                    correct_text = item["options"][item["answer"]]
                    ok = chosen == correct_text
                    score += int(ok)
                    review.append({
                        "Item": i,
                        "Question": item["q"],
                        "Your answer": chosen,
                        "Correct answer": correct_text,
                        "Result": "Correct" if ok else "Wrong",
                        "Explanation": item["why"],
                    })
                st.session_state["exam_result"] = {
                    "name": name.strip(),
                    "section": section.strip(),
                    "score": score,
                    "total": len(EXAM),
                    "review": review,
                    "short": list(zip(SHORT_ANSWER, short)),
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                st.rerun()
    else:
        pct = result["score"] / result["total"] * 100
        st.success(f"Submitted by {result['name']} on {result['time']}")
        m1, m2 = st.columns(2)
        m1.metric("Part I score", f"{result['score']} / {result['total']}")
        m2.metric("Percentage", f"{pct:.0f}%")

        st.markdown("### Review of Part I")
        for r in result["review"]:
            icon = "✅" if r["Result"] == "Correct" else "❌"
            with st.expander(f"{icon} Item {r['Item']}: {r['Question']}"):
                st.write(f"**Your answer:** {r['Your answer']}")
                st.write(f"**Correct answer:** {r['Correct answer']}")
                st.write(f"**Why:** {r['Explanation']}")

        st.markdown("### Part II - Your short answers")
        for q, a in result["short"]:
            st.write(f"**{q}**")
            st.write(a if a.strip() else "_(no answer)_")

        # Downloadable record
        rows = []
        for r in result["review"]:
            rows.append({
                "Name": result["name"], "Section": result["section"],
                "Submitted": result["time"], "Part": "I",
                "Item": r["Item"], "Question": r["Question"],
                "Answer": r["Your answer"], "Correct answer": r["Correct answer"],
                "Result": r["Result"],
            })
        for j, (q, a) in enumerate(result["short"], start=1):
            rows.append({
                "Name": result["name"], "Section": result["section"],
                "Submitted": result["time"], "Part": "II",
                "Item": j, "Question": q, "Answer": a,
                "Correct answer": "", "Result": "For manual grading",
            })
        record = pd.DataFrame(rows)
        st.download_button(
            "Download exam record (CSV)",
            record.to_csv(index=False),
            f"midterm_{result['name'].replace(' ', '_')}.csv",
            "text/csv",
        )

        if st.button("Start a new attempt"):
            for k in list(st.session_state.keys()):
                if k.startswith("mcq_") or k.startswith("sa_") or k == "exam_result":
                    del st.session_state[k]
            st.rerun()
