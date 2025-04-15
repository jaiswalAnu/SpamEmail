import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Ensure the necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    # Step 1: Lowercase the text
    text = text.lower()
    # Step 2: Tokenize the text
    text = nltk.word_tokenize(text)

    # Step 3: Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    # Step 4: Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Step 5: Apply stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# Load the pre-trained vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit interface
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input text
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize the transformed text
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict the class of the input
    result = model.predict(vector_input)[0]
    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
