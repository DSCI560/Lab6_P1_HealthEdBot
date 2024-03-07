import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import torch

""" PDF Extraction"""
def extract_text_from_pdf_folder(pdf_path):
    text_data = []
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()
        text_data.append(text)

    return text_data


""" Text Chunk"""
def split_text_into_chunks(text_data):
    if isinstance(text_data, str):
        splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100,length_function=len)
        chunks = splitter.split_text(text_data)
        return chunks
    else:
        raise ValueError("text_data should be a string")

""" Vector DataStore"""
def create_vector_datastore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore



def get_user_question():
    user_input = input("Question: ")
    if user_input.lower() != 'exit':
      user_input_2 = 100
      return user_input, user_input_2
    return user_input, 0


def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key="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",
     max_tokens=1000)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
        max_tokens_limit=4000
    )
    return conversation_chain

def main():

    # PDF Extraction
    extracted_text = extract_text_from_pdf_folder("Ads cookbook .pdf")
    print("PDF Extraction Completed")

    # Text Chunk
    text_chunks = split_text_into_chunks(str(extracted_text))
    print("Text Chunk Completed")


    # Vector _Datastore
    vector_datastore = create_vector_datastore(text_chunks)  # Assuming you have the text chunks
    print("Vectorizing Completed")


    # Conversation Chain
    conversation_chain = create_conversation_chain(vector_datastore)

    while True:
        user_question, max_length = get_user_question()
        if user_question.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break

        response = conversation_chain({'question': user_question})
        print("AI: ",response['answer'])

if __name__ == "__main__":
    main()
