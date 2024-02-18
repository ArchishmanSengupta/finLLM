# custom class for splitting text into smaller chunks.
from langchain.text_splitter import RecursiveCharacterTextSplitter
# loaders for PDF documents and directories
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# wrapper for using Hugging Face models to generate text embeddings
from langchain.embeddings import HuggingFaceEmbeddings
# Facebook AI Similarity Search (Faiss), a library that allows us to quickly search for multimedia documents
# Great article on FAISS: https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/
from langchain.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

# create a vector db
def create_vector_database():
  # using loader class for PDF loading
  loader = DirectoryLoader(DATA_PATH, glob = '*.pdf',loader_cls = PyPDFLoader)
  # Loading the list of doc objects into documents
  documents = loader.load()
  # Chunking parameters size of 500 characters with 50 overlaps(so that no infomation is lost)
  text_splitters = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
  # Chunking the document based on the parameters given
  texts = text_splitters.split_documents(documents)
  
  # generating sentence embeddings, using the MiniLM architecture with 6 layers running on CPU
  embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-V2', model_kwargs = {'device':'cpu'})
  # create the vector db using the embeddings
  db = FAISS.from_documents(texts, embeddings)
  # store in local
  db.save_local(DB_FAISS_PATH)
  
if __name__ == '__main__':
  create_vector_database()