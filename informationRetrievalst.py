import streamlit as st
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
import transformers
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from transformers import AutoModelForSeq2SeqLM
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_chat import message

st.header("Information Retrieval from PDF file",divider="gray")

class RetrievalApplicaltion:
    def get_file_path(uploaded_file):
        cwd = os.getcwd()
        temp_dir = os.path.join(cwd, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    
    f = st.file_uploader("Question Answering on RHP of IPO", type=(["pdf"]))
    if f is not None:
        path_in = get_file_path(f)
        print("*"*10,path_in)
    else:
        path_in = None
    
    prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")

    if "model" not in st.session_state:
        model_id = 'google/flan-t5-large'
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        pipe = pipeline("text2text-generation",model=model, tokenizer=tokenizer, max_length=100)
        local_llm = HuggingFacePipeline(pipeline=pipe,model_kwargs = {'temperature':0.6})
        st.session_state["model"] = local_llm
    
    if "vectordb" not in st.session_state and path_in:
        loader=PyPDFLoader(file_path=path_in)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(pages)
        embeddingsTinysts = HuggingFaceEmbeddings(model_name="sergeyzh/rubert-tiny-sts")
        persist_directory = 'db'
        vectordb = Chroma.from_documents(documents=texts,embedding=embeddingsTinysts, persist_directory=persist_directory)
        vectordb.persist()
        vectordb=None
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddingsTinysts)
        st.session_state["vectordb"] = vectordb
    
    def answer(query,new_vectorstore):
        llm_model = st.session_state["model"]
        retriever = new_vectorstore.as_retriever(search_kwargs={"k": 2})
        qa_chain = RetrievalQA.from_chain_type(llm=llm_model, chain_type="stuff", retriever=retriever, return_source_documents=True)
        llm_response = qa_chain(query)
        return llm_response['result']
    
    if prompt and path_in:
        with st.spinner("Generating response.."):
            new_vectorstore=st.session_state["vectordb"]
            generated_response = answer(query=prompt,new_vectorstore=new_vectorstore)
            message(generated_response)
        
    
        
       
    


    






if __name__ == "__main__":
    RetrievalApplicaltion()