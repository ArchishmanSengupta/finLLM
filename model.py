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