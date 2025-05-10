import streamlit as st
import pandas as pd
import joblib
import os

# Load models and vectorizer (from current directory)
LR = joblib.load("logistic_regression_model.pkl")
DT = joblib.load("decision_tree_model.pkl")
GB = joblib.load("gradient_boosting_model.pkl")
RF = joblib.load("random_forest_model.pkl")
vectorization = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing function
def wordopt(text):
    import re
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = text.lower()
    return text

def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

def predict_news(news):
    df = pd.DataFrame({'text': [news]})
    df['text'] = df['text'].apply(wordopt)
    new_xv_test = vectorization.transform(df['text'])

    predictions = {
        "Logistic Regression": output_label(LR.predict(new_xv_test)[0]),
        "Decision Tree": output_label(DT.predict(new_xv_test)[0]),
        "Gradient Boosting": output_label(GB.predict(new_xv_test)[0]),
        "Random Forest": output_label(RF.predict(new_xv_test)[0]),
    }
    return predictions

# Streamlit UI
st.title("📰 Fake News Detection System")
st.write("Enter the news content below to check if it's **Fake** or **Not Fake** using multiple ML models.")

news_input = st.text_area("📝 Paste your news article here:", height=300)

if st.button("Detect"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text first.")
    else:
        with st.spinner("Analyzing the news..."):
            results = predict_news(news_input)
            st.success("✅ Prediction complete!")

            for model, prediction in results.items():
                st.write(f"**{model} Prediction:** {prediction}")
