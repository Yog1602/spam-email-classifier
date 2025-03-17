import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\Yog Dalal\Desktop\Email using logistic legression\mail_data.csv")

# Rename columns if necessary
df.rename(columns={'Category': 'label', 'Message': 'text'}, inplace=True)

# Convert labels to binary (0 = Not Spam, 1 = Spam)
df['label'] = df['label'].map({"ham": 0, "spam": 1})

# Drop any NaN values
df.dropna(inplace=True)

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        return text
    return ""

# Apply preprocessing
df['text'] = df['text'].apply(preprocess_text)

# Ensure dataset is not empty
if df.empty:
    raise ValueError("Dataset is empty after preprocessing.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_tfidf, y_train)

# Model evaluation
y_pred = classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Streamlit UI
st.title("Spam Email Classifier (Logistic Regression)")
st.write("Enter an email below to check if it's Spam or Not Spam.")

# ---- Dataset Preview ----
st.subheader("Dataset Overview")
st.write(df.head(10))  # Show first 10 rows of the dataset

# ---- Model Performance ----
st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")
st.text("Classification Report:")
st.text(report)

# ---- User Input for Classification ----
user_input = st.text_area("Enter email text:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess and predict
        processed_input = preprocess_text(user_input)
        input_tfidf = vectorizer.transform([processed_input])
        prediction = classifier.predict(input_tfidf)[0]

        # Display result
        if prediction == 1:
            st.error("This email is SPAM!")
        else:
            st.success("This email is Not Spam.")
