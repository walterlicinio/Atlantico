# A streamlit data app that receives a pdf input and outputs the tokenized and lemmatized text
import string
import streamlit as st
import nltk 
import pdfplumber
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('word_tokenize')
nltk.download('omw-1.4')

print("oi")
st.title("Streamlit Data App")
st.header("Tokenize and Lemmatize")

pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf:
    text = ''
    with pdfplumber.open(pdf) as pdf_file:
        for page in pdf_file.pages:
            text += page.extract_text()
    text = text.replace('\n', ' ')
    # remove all punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    st.header("Original Text")
    st.write('First 1000 chars:')
    st.write(text[:1000])   
    
    # decapitalize all words
    text = text.lower() 
    
    text_tokenized = nltk.word_tokenize(text)
    st.header('Tokenized text')  
    st.write(text_tokenized)
   
    lemmatizer = nltk.stem.WordNetLemmatizer()
    text_lemmatized = [lemmatizer.lemmatize(w) for w in text_tokenized]
 
    
    # remove all stopwords portuguese
    stopwords = nltk.corpus.stopwords.words('portuguese')
    st.write(stopwords)
    text_lemmatized = [w for w in text_lemmatized if w not in stopwords]
    st.header('Lemmatized text')  
    st.write(text_lemmatized)
    
    st.header('Plotting the frequency of the lemmatized text')
    freq = nltk.FreqDist(text_lemmatized)
    freq.plot(20)
    plt.show()
    st.pyplot()
 