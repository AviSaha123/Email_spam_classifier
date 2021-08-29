import streamlit as st 
import pickle 
from nltk.corpus import stopwords
import string
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    x = []
    for i in text:
        if i.isalnum():
            x.append(i)

    text = x[:]
    x.clear()

    for j in text:
        if j not in stopwords.words('english') and j not in string.punctuation:
            x.append(j)

    text = x[:]
    x.clear()

    for i in text:
        x.append(ps.stem(i))

    return ' '.join(x)


tfid = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email Spam Classifier")
input_message = st.text_area("Enter Your Message")
if st.button('Predict'):
    
    ##Preprocessing
    transformed_text = text_transform(input_message)
    # Text Vectorization
    vector_input = tfid.transform([transformed_text])
    # Model Prdeiction 
    result = model.predict(vector_input)[0]
    #Display 

    if result == 1:
        st.header('Spam')
    else: 
        st.header('Not Spam') 
    

