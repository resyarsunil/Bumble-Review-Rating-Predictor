import streamlit as st
import joblib
from Automated_Review_System.Imbalanced_data.imbalanced_UI.preprocess import preprocess

# -----------------------
# Load Saved Models
# -----------------------
vectorizer = joblib.load("vectorizer.pkl")

lr_model = joblib.load("logistic_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
svm_model = joblib.load("svm_model.pkl")

# -----------------------
# Page
# -----------------------
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

model = st.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "Random Forest",
        "Linear SVM"
    ]
)

if st.button("Predict Rating"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        processed_review = preprocess(review)

        review_vector = vectorizer.transform([processed_review])

        if model == "Logistic Regression":
            prediction = lr_model.predict(review_vector)[0]

        elif model == "Random Forest":
            prediction = rf_model.predict(review_vector)[0]

        else:
            prediction = svm_model.predict(review_vector)[0]

        st.success(f"⭐ Predicted Rating: {prediction} Star")

        st.subheader("Processed Review")

        st.write(processed_review)