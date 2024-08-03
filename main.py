__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
#import chromadb
import sqlite3
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

# Fichero .txt de entrada
# Formato de archivo
# Dividir archivo
# Crear incrustaciones
# Almacenar incrustaciones en el almacén de vectores
# Consulta de entrada
# Ejecutar la cadena de control de calidad
# Salida

def generate_response(file, openai_api_key, query):
    #formato de archivo
    reader = PdfReader(file)
    formatted_document = []
    for page in reader.pages:
        formatted_document.append(page.extract_text())
    #dividir archivo
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    docs = text_splitter.create_documents(formatted_document)
    #crear incrustaciones
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    #cargar en base de datos vectorial
    #store = Chroma.from_documents(texts, embeddings)

    store = FAISS.from_documents(docs, embeddings)
    
    #crear cadena de recuperación
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=store.as_retriever()
    )
    #ejecutar cadena con consulta
    return retrieval_chain.run(query)

st.set_page_config(
    page_title="Preguntas y respuestas de un largo documento PDF"
)
st.title("Preguntas y respuestas de un largo documento PDF")

st.write("Contacte con [Matias Toro Labra](https://www.linkedin.com/in/luis-matias-toro-labra-b4074121b/) para construir sus proyectos de IA")

uploaded_file = st.file_uploader(
    "Cargar un documento .pdf",
    type="pdf"
)

query_text = st.text_input(
    "Escriba su pregunta:",
    placeholder="Escriba aquí su pregunta",
    disabled=not uploaded_file
)

result = []
with st.form(
    "myform",
    clear_on_submit=True
):
    openai_api_key = st.text_input(
        "OpenAI API Key:",
        type="password",
        disabled=not (uploaded_file and query_text)
    )
    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_file and query_text)
    )
    if submitted and openai_api_key.startswith("sk-"):
        with st.spinner(
            "Espera, por favor. Estoy trabajando en ello..."
            ):
            response = generate_response(
                uploaded_file,
                openai_api_key,
                query_text
            )
            result.append(response)
            del openai_api_key
            
if len(result):
    st.info(response)