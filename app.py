import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Load saved vectorizer and model
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Preprocessing Functions
STOPWORDS = set(stopwords.words('english'))
punctuations_list = string.punctuation
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    # Remove punctuations
    text = text.translate(str.maketrans('', '', punctuations_list))
    # Remove repeating characters
    text = re.sub(r'(.)\1+', r'\1', text)
    # Tokenize
    tokens = tokenizer.tokenize(text)
    # Stemming
    tokens = [stemmer.stem(token) for token in tokens]
    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

# Streamlit App
st.title("Fake News Detection App")
st.write("Enter a news article below to check if it's real or fake.")

# Input Text Box
user_input = st.text_area("News Article", "")

if st.button("Predict"):
    if user_input.strip():
        # Preprocess input text
        cleaned_text = clean_text(user_input)
        # Vectorize the text
        vectorized_text = vectorizer.transform([cleaned_text])
        # Predict using the loaded model
        prediction = model.predict(vectorized_text)
        prediction_prob = model.predict_proba(vectorized_text)

        # Display Results
        if prediction[0] == 1:
            st.success(f"The article is classified as **Real News**.")
        else:
            st.error(f"The article is classified as **Fake News**.")
        
        # Show probabilities
        st.write(f"Prediction Confidence:")
        st.write(f"- Real News: {prediction_prob[0][1] * 100:.2f}%")
        st.write(f"- Fake News: {prediction_prob[0][0] * 100:.2f}%")
    else:
        st.warning("Please enter some text for prediction.")
