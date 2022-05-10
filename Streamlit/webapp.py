# A streamlit data app that receives a pdf input and outputs the tokenized and lemmatized text

# Imports & NLTK Downloads
import string
import streamlit as st
import nltk 
import pdfplumber
import matplotlib.pyplot as plt
import regex as re
import pandas as pd

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('word_tokenize')
nltk.download('omw-1.4')

# Streamlit Global
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Atlantico - Data App - Tokenizer")
st.header("Tokenizar e Lematizar")
st.write("Envie um arquivo PDF para ser tokenizado, lemmatizado e analisado.")
pdf = st.file_uploader("Upload de arquivo pdf.", type=["pdf"])

if pdf:
    text = ''
    with pdfplumber.open(pdf) as pdf_file:
        for page in pdf_file.pages:
            text += page.extract_text()
    text = text.replace('\n', ' ')   
    # Remove all punctuation using re sub
    text = re.sub(r'[^\w\s]', '', text)
    # Remove all digits using re sub
    text = re.sub(r'\d+', '', text)
    st.header("Texto original")
    st.write('**Primeiros mil caracteres abaixo**:')
    st.write(text[:1000])   
    
    # decapitalize all words
    text = text.lower() 
        
    text_tokenized = nltk.word_tokenize(text)
    st.header('Texto Tokenizado')  
    st.write(text_tokenized)
   
    lemmatizer = nltk.stem.WordNetLemmatizer()
    text_lemmatized = [lemmatizer.lemmatize(w) for w in text_tokenized]
 
    
    # Remove all stopwords in portuguese
    stopwords = nltk.corpus.stopwords.words('portuguese')
    text_lemmatized = [w for w in text_lemmatized if w not in stopwords]
    st.header('Texto Lematizado')  
    st.write(text_lemmatized)
    
    # Term Frequency
    st.header('Frequência de Termos')
    st.write('### Gráfico de frequência de termos')
    freq = nltk.FreqDist(text_lemmatized)
    freq.plot(20)
    plt.show()
    st.pyplot()

     
    # Generate a table with the column being the word and the row being the document frequency
    st.write("### Tabela de frequência de termos")
    st.dataframe(freq.most_common(), width=800, cols=['term', 'freq'])
    st.pyplot()
    
    

    
    
 