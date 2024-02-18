import asyncio
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# generating prompt templates
from langchain import PromptTemplate
# wrapper for using Hugging Face models to generate text embeddings
from langchain.embeddings import HuggingFaceEmbeddings
# Facebook AI Similarity Search (Faiss), a library that allows us to quickly search for multimedia documents
# Great article on FAISS: https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/
from langchain.vectorstores import FAISS
# Wrapper for using conditional transformers (CTransformers) for natural language processing tasks
from langchain.llms import CTransformers
# Functionality for building retrieval-based question answering systems
from langchain.chains import RetrievalQA
# Additional package for chain-based NLP tasks
import chainlit as cl


DB_FAISS_PATH = 'vectorstores/db_faiss'

custom_prompt_template = """
You're tasked with providing a helpful response based on the given context and question.
Accuracy is paramount, so if you're uncertain, it's best to acknowledge that rather than providing potentially incorrect information.

Context: {context}
Question: {question}


Please craft a clear and informative response that directly addresses the question.
Aim for accuracy and relevance, keeping the user's needs in mind.
Response:
"""
# setting up a pipeline for question-answering (QA) retrieval
def set_custom_prompt():
    # Prompt template for QA retrieval for each vector stores
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

"""
This method initializes a question-answering (QA) retrieval chain, which consists of a conditional
    transformer model (CTransformers) for generating responses, a prompt template for constructing
    prompts based on user queries, and a vector store (FAISS) for retrieving relevant documents
    to support the QA process.

    - llm (CTransformers): The conditional transformer model (CTransformers) plays a central role
    in generating responses to user queries. It leverages pre-trained language
    models to understand user questions and produce informative answers.

    - prompt (PromptTemplate): The prompt template (PromptTemplate) defines the structure of prompts
    provided to the QA model. It includes placeholders for the context
    and question, guiding the model on how to interpret and respond to
    user queries effectively.

    - db (FAISS): The vector store (FAISS) serves as a repository of documents or passages that
    the QA model can search through to find relevant information. It enables efficient
    retrieval of documents based on the similarity of their embeddings to the user query.

    The method initializes a RetrievalQA object, which orchestrates the interaction between the
    conditional transformer model, prompt template, and vector store to facilitate QA retrieval.
    The RetrievalQA object is configured with the specified components, including the model,
    prompt template, retriever (vector store), and additional settings for controlling the QA process.

    The initialized RetrievalQA object is returned, ready to be used for responding to user queries
    by retrieving relevant information from the vector store and generating informative answers
    using the conditional transformer model.
"""
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

# loads a conditional transformer model for the QA retrieval task on llama-2-7b-chat-ggml
def load_llm():
    llm = CTransformers(
        model = 'models/llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# QA Model Function
async def setup_qa_bot():
    #Step 1: Initialize Hugging Face embeddings
    hugging_face_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-V2', model_kwargs={'device': 'cpu'})

    # Step 2: Load FAISS vector store with embeddings
    faiss_db = FAISS.load_local(DB_FAISS_PATH, embeddings=hugging_face_embeddings)

    # Step 3: Load conditional transformer model (LLAMA-2)
    llama_model = load_llm()

    # Step 4: Set up custom prompt for QA retrieval
    qa_prompt = set_custom_prompt()

    # Step 5: Set up QA retrieval chain
    qa_retrieval_chain = retrieval_qa_chain(llama_model, qa_prompt, faiss_db)

    return qa_retrieval_chain

# Output function
async def final_response(query):
    qa_result = await setup_qa_bot()
    response = await qa_result({'query': query})
    return response

# chainlit code
@cl.on_chat_start
async def start():
    chain = await setup_qa_bot()
   #Send a message to indicate initialization
    init_message = cl.Message(content="Initializing the system.")
    await init_message.send()
    
    # Update the message to welcome the user and prompt for questions
    init_message.content = 'Welcome to the Encyclopedia of Finance. Please ask your questions.'
    await init_message.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()

if __name__ == "__main__":
    asyncio.run(cl.main())
