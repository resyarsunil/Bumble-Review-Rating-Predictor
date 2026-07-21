import streamlit as st
import re
import joblib
import base64
from pathlib import Path
from preprocess import preprocess


st.set_page_config(
    page_title="Bumble Review Rating Predictor",
    page_icon="💛",
    layout="centered"
)


# LOAD MODELs

@st.cache_resource
def load_models():
    vectorizer = joblib.load("vectorizer.pkl")
    lr_model = joblib.load("logistic_model.pkl")
    svm_model = joblib.load("svm_model.pkl")
    return vectorizer, lr_model, svm_model


vectorizer, lr_model, svm_model = load_models()


# BACKGROUND IMAGE

IMAGE_PATH = Path("yellow.png")

def get_base64(image_path):
    with open(image_path, "rb") as image:
        return base64.b64encode(image.read()).decode()


if IMAGE_PATH.exists():

    encoded_image = get_base64(IMAGE_PATH)

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <style>
        .stApp{
            background-color:#FFC629;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# TRANSPARENT WHITE CONTENT BOX
# This wraps ALL the page content (title, text area, selectbox, button, results)

st.markdown(
    """
    <style>
    .block-container {
        background-color: #2D1B12;
        border-radius: 20px;
        padding: 15px 45px 40px 45px;
        margin-top: 0px;
        margin-bottom: 30px;
        box-shadow: 0 12px 45px rgba(0, 0, 0, 1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# TITLE

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@800&display=swap');

.title {
    font-family: 'Poppins', sans-serif;
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #FFFFFF;
    margin-bottom: 8px;
}
</style>

<h1 class="title">💛 Bumble Review Rating Predictor</h1>
""", unsafe_allow_html=True)


# SUBTITLE

st.markdown("""
<p style="
text-align:center;
font-size:35px;
font-weight:bold;
color:#FFD54F;">
Predict the rating of a Bumble review using Machine Learning
</p>
""", unsafe_allow_html=True)


st.divider()


st.markdown(
    "<p style='font-size:22px; color:#FFF8E7; font-weight:bold;'>Enter a Bumble Review</p>",
    unsafe_allow_html=True
)
review = st.text_area(
    "Review",
    placeholder="Type your review here...",
    label_visibility="collapsed",
    height=180
)

st.markdown(
    "<p style='font-size:22px; color:#FFF8E7;font-weight:bold;'>Choose a Machine Learning Model</p>",
    unsafe_allow_html=True
)
model = st.selectbox(
    "Machine Learning Model",
    [
        "Logistic Regression",
        "Linear SVM"
    ]
)


# PREDICT

if st.button("Predict Rating"):

    review = review.strip()

    if review == "":
        st.markdown(
            "<p style='font-size:20px; color:orange;'>⚠️ Please enter a review.</p>",
            unsafe_allow_html=True
        )

    elif review.isdigit():
        st.markdown(
            "<p style='font-size:20px; color:red;'>❌ Invalid review. Please enter meaningful text.</p>",
            unsafe_allow_html=True
        )

    elif len(set(review.replace(" ", "").lower())) == 1:
        st.markdown(
            "<p style='font-size:20px; color:red;'>❌ Invalid review. Please enter meaningful text.</p>",
            unsafe_allow_html=True
        )

    elif len(re.findall(r"[A-Za-z]{2,}", review)) < 2:
        st.markdown(
            "<p style='font-size:20px; color:red;'>❌ Invalid review. Please enter meaningful text.</p>",
            unsafe_allow_html=True
        )

    else:

        processed_review = preprocess(review)

        if processed_review.strip() == "":
            st.markdown(
                "<p style='font-size:20px; color:red;'>❌ Invalid review after preprocessing.</p>",
                unsafe_allow_html=True
            )

        else:

            review_vector = vectorizer.transform([processed_review])

            if model == "Logistic Regression":
                prediction = lr_model.predict(review_vector)[0]


            else:
                prediction = svm_model.predict(review_vector)[0]

            # Display stars based on predicted rating
            stars = "⭐" * int(prediction)

            st.markdown(
                f"""
                <p style='font-size:26px; font-weight:bold; color:green;'>
                    Predicted Rating: {prediction} {stars}
                </p>
                """,
                
                unsafe_allow_html=True
            )

            st.markdown(
                "<p style='font-size:22px; font-weight:bold;'>Processed Review</p>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<p style='font-size:18px;'>{processed_review}</p>",
                unsafe_allow_html=True
            )