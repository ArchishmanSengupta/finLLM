# generating prompt templates
from langchain import PromptTemplate
# wrapper for using Hugging Face models to generate text embeddings
from langchain.embeddings import HuggingFaceEmbeddings
# Facebook AI Similarity Search (Faiss), a library that allows us to quickly search for multimedia documents
# Great article on FAISS: https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/
from langchain.vectorestores import FAISS
# Wrapper for using conditional transformers (CTransformers) for natural language processing tasks
from langchain.llms import CTransformers
# Functionality for building retrieval-based question answering systems
from langchain.chains import RetrievalQA
# Additional package for chain-based NLP tasks
import chainlit as cl

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """
You're tasked with providing a helpful response based on the given context and question.
Accuracy is paramount, so if you're uncertain, it's best to acknowledge that rather than providing potentially incorrect information.

Context: {}
Question: {question}


Please craft a clear and informative response that directly addresses the question.
Aim for accuracy and relevance, keeping the user's needs in mind.
Response:
"""