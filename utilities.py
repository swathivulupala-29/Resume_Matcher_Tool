#'this func extract text from pdf file and return text'
import fitz
import numpy as np

# extract data from file 
def extract_text_from_pdf(pdf_path):
    'this func extract text from pdf file and return text'
    doc=fitz.open(pdf_path)
    text=''
    for page in doc:
        text+=page.get_text("text")
    return text


# embedding function to create vectors
def get_embedding(text):
    if not isinstance(text, str) or text.strip() == "":
        raise ValueError("Invalid input: Expected a non-empty string for embedding.")

    embedding = model.encode([text])

    # Ensure it's a valid NumPy array
    if isinstance(embedding, np.ndarray) and embedding.ndim > 0:
        return embedding[0]  # Get the first vector
    elif isinstance(embedding, np.float32):  
        return np.array([embedding])  # Convert float to array
    else:
        raise ValueError("Unexpected embedding format.")
    
    
# dependence libraries for cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
model=SentenceTransformer("paraphrase-MiniLM-L6-v2")


# cosine similarity function
def compute_similarity(resume_text,job_text):
    '''this function checks cosine similarity by using cosine function'''
    resume_embedding=get_embedding(resume_text)
    job_embedding=get_embedding(job_text)
    similarity_score=cosine_similarity([resume_embedding],[job_embedding]),[0][0]

    return similarity_score