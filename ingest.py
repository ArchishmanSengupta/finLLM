# custom class for splitting text into smaller chunks.
from langchain.text_splitters import RecursiveCharacterTextSplitter
# loaders for PDF documents and directories
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# wrapper for using Hugging Face models to generate text embeddings
from langchain.embeddings import HuggingFaceEmbeddings
# Facebook AI Similarity Search (Faiss), a library that allows us to quickly search for multimedia documents
from langchain.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"