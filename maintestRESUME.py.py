from utilities import get_embedding, extract_text_from_pdf, compute_similarity
from streamlit import streamlit
import streamlit as st
import keras
import tensorflow
import tf_keras as keras
import gensim
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pymupdf
import fitz
from sklearn.metrics.pairwise import cosine_similarity
import time


st.title('NLP -Resume Matching Tool')

# File uploader (drag & drop or browse)
uploaded_file = st.file_uploader("Drag a Resume and drop here, or click to browse", 
                                 type=["csv", "txt", "pdf", "docx"])


resume_text =  extract_text_from_pdf(uploaded_file)
#resume_text =  extract_text_from_pdf('Sr.Python_Developer (3).docx')

resume_embedding = get_embedding("resume_text")

#jd=st.text_area("enter your job discrtiption here")
jd=st.text_input("enter your job discrtiption here:", placeholder="Type here...")

st.write('Check below : if JD Matching or Not ')

value = compute_similarity(resume_text, jd)


if st.button("Match Now"):
    st.write("Matching Process Started! 🚀")


progress_bar = st.progress(0)  
for percent_complete in range(101):  
    time.sleep(0.05)  
    progress_bar.progress(percent_complete / 100) 


st.success("Process Completed!")  


if value[0] >= 0.65:
  st.success('Result:   Resume Matched with the Job Description')
else:
  st.warning('Result:   Resume Do Not Matched with the Job Description')



















