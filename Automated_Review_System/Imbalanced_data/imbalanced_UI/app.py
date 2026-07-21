import streamlit as st
import re
import joblib
from pathlib import Path
from preprocess import preprocess

# Load Saved Models
base_dir = Path(__file__).resolve().parent

vectorizer = joblib.load(base_dir / "vectorizer.pkl")
lr_model = joblib.load(base_dir / "logistic_model.pkl")
svm_model = joblib.load(base_dir / "svm_model.pkl")

rf_model = None
rf_model_path = base_dir / "random_forest_model.pkl"
if rf_model_path.exists():
    rf_model = joblib.load(rf_model_path)

st.set_page_config(
    page_title="Bumble Review Rating Predictor",
    page_icon="💛",
    layout="centered"
)

st.title("💛 Bumble Review Rating Predictor")
st.write("Predict the rating of a Bumble review using Machine Learning.")
st.divider()

review = st.text_area(
    "Enter a Bumble Review",
    height=180
)

available_models = ["Logistic Regression", "Linear SVM"]
if rf_model is not None:
    available_models.append("Random Forest")

model = st.selectbox("Choose Model", available_models)

if st.button("Predict Rating"):
    review = review.strip()

    if review == "":
        st.warning("Please enter a review.")

    elif review.isdigit():
        st.error("❌ Invalid review. Please enter a meaningful review.")

    elif len(set(review.replace(" ", "").lower())) == 1:
        st.error("❌ Invalid review. Please enter a meaningful review.")

    elif len(re.findall(r"[A-Za-z]{2,}", review)) < 2:
        st.error("❌ Invalid review. Please enter a meaningful review.")

    else:
        processed_review = preprocess(review)

        if processed_review.strip() == "":
            st.error("❌ Invalid review. Please enter a meaningful review.")

        else:
            review_vector = vectorizer.transform([processed_review])

            if model == "Logistic Regression":
                prediction = lr_model.predict(review_vector)[0]
            elif model == "Random Forest":
                if rf_model is None:
                    st.error("❌ Random Forest model is not available.")
                    st.stop()
                prediction = rf_model.predict(review_vector)[0]
            else:
                prediction = svm_model.predict(review_vector)[0]

            st.success(f"⭐ Predicted Rating: {prediction} Star")
            st.subheader("Processed Review")
            st.write(processed_review)