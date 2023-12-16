import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
       pdf_reader = PdfReader(pdf)
       for page in pdf_reader.pages:
           text += page.extract_text()
    return text  


def get_text_chunks(raw_text):

    text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=300,
            length_function=len            
        ) 
    chunks=text_splitter.split_text(raw_text)
    
    return chunks

def get_vectorstore(chunks_text):

    embeddings = OpenAIEmbeddings()
    vectorstore =  FAISS.from_texts(texts=chunks_text,embedding=embeddings)
    
    return vectorstore    


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Single PDF",page_icon=":books")
    st.header("Chat with multiple PDFs:books:")
    st.text_input("Ask a question about your documents:")
    
    with st.sidebar:
      st.subheader("Your documents")
      pdf_docs=st.file_uploader("Upload your pdfs here and click on 'Process'",accept_multiple_files=True,type="pdf")
      if st.button("Process"):
        with st.spinner("Processing"): 
          # get pdf text
            raw_text = get_pdf_text(pdf_docs) 
    
           # get the text chunks
            chunks_text=get_text_chunks(raw_text)
        
           # create embeddings /vector store
            vector_store= get_vectorstore(chunks_text)
            
           # get vectorstore
           
        
        
        user_question = st.text_input("Ask a question about your PDF:")
        
        if user_question:
            docs=knowledge_base.similarity_search(user_question)
                
        llm = OpenAI()
        chain = load_qa_chain(llm,type="stuff")
        
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs,question=user_question)
            st.write(response)
            
        st.write(response)
        
        
if __name__=='__main__':
    main()