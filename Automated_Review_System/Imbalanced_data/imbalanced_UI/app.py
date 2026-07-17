import streamlit as st
import re
import joblib
from preprocess import preprocess


# Load Saved Models

vectorizer = joblib.load("vectorizer.pkl")

lr_model = joblib.load("logistic_model.pkl")
svm_model = joblib.load("svm_model.pkl")



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

    review = review.strip()

    if review == "":
        st.warning("Please enter a review.")

    # Reject reviews containing only numbers
    elif review.isdigit():
        st.error("❌ Invalid review. Please enter a meaningful review.")

    # Reject reviews with only one repeated character (AAAAAA, 111111, !!!!!)
    elif len(set(review.replace(" ", "").lower())) == 1:
        st.error("❌ Invalid review. Please enter a meaningful review.")

    # Reject reviews with no valid words
    elif len(re.findall(r"[A-Za-z]{2,}", review)) < 2:
        st.error("❌ Invalid review. Please enter a meaningful review.")

    else:

        processed_review = preprocess(review)

        # If preprocessing removes everything
        if processed_review.strip() == "":
            st.error("❌ Invalid review. Please enter a meaningful review.")

        else:

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