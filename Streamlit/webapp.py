# A streamlit data app that receives a pdf input and outputs the tokenized and lemmatized text

# Imports & NLTK Downloads
import string
import math
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
st.title("Análise de Texto - NLP")
# st.header("Desenvolvido por Walter Licínio")
st.write("Envie um ou mais arquivos PDF para serem tokenizados, lemmatizados e analisados.")
pdf_list = st.file_uploader("Upload de arquivo pdf.", type=["pdf"], accept_multiple_files=True)
st.write("Para resetar a aplicação, remova os arquivos.")


all_lemmatized_docs=[]    

def tokenizer(pdf_list):
    if pdf_list:   
        for pdf  in pdf_list:
            text = ''
            with pdfplumber.open(pdf) as pdf_file:
                for page in pdf_file.pages:
                    text += page.extract_text()
            text = text.replace('\n', ' ')   
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            st.header("Arquivo: "+pdf.name)
            st.write('**Primeiros 200 caracteres abaixo**:')
            st.write(text[:100] + "...")   
        
            text = text.lower()         
            text_tokenized = nltk.word_tokenize(text)
        
            lemmatizer = nltk.stem.WordNetLemmatizer()
            text_lemmatized = [lemmatizer.lemmatize(w) for w in text_tokenized]
        
            stopwords = nltk.corpus.stopwords.words('portuguese')
            text_lemmatized = [w for w in text_lemmatized if w not in stopwords]
    
            
            for word in text_lemmatized:
                all_lemmatized_docs.append(word)
            
            # Term Frequency
            st.header('TF - Frequência de Termos - '+ pdf.name )
            # st.write('### Gráfico de frequência de termos - '+ pdf.name)
            freq = nltk.FreqDist(text_lemmatized)
            # freq.plot(30)
            # plt.show()
            # st.pyplot()
            st.dataframe(freq.most_common())
            st.markdown("""---""")
  

def doc_frequency(pdf_list):  
    if pdf_list:
        st.write("# DF - Document Frequency")
        st.write("Ocorrência de cada termo no conjunto de documentos.")
        freq = nltk.FreqDist(all_lemmatized_docs)
        st.dataframe(freq.most_common())
        st.markdown("""---""")
        
def inverse_document_frequency(pdf_list):
    if pdf_list:
        st.write("# IDF - Inverse Document Frequency")
        st.write("Onde o valor de IDF é o logaritmo da divisão entre o número de documentos e o número de documentos em que o termo aparece.")
        freq = nltk.FreqDist(all_lemmatized_docs)
        idf = {}
        for word in freq.keys():
            idf[word] = math.log(len(pdf_list) / (freq[word]+1))
        idf_df = pd.DataFrame.from_dict(idf, orient='index')
        st.dataframe(idf_df)
        st.markdown("""---""")
    
tokenizer(pdf_list)
doc_frequency(pdf_list)
inverse_document_frequency(pdf_list)

    
    
 