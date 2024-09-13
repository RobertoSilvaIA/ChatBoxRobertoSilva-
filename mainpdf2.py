import streamlit as st
from langchain.chains.summarize.map_reduce_prompt import prompt_template
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


openai_api_key = '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'


# Función para procesar los archivos PDF subidos
def actualizar_embeddings(archivos_subidos):
    todos_los_fragmentos = []

    for archivo in archivos_subidos:
        # Guardar el archivo subido temporalmente en el disco
        with open(archivo.name, "wb") as f:
            f.write(archivo.getbuffer())

        # Cargar el archivo PDF desde el archivo guardado temporalmente
        loader = PyPDFLoader(archivo.name)
        docs = loader.load()

        # Dividir el texto en fragmentos manejables
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
        chunked_documents = text_splitter.split_documents(docs)

        # Agregar los fragmentos de cada documento a la lista
        todos_los_fragmentos.extend(chunked_documents)

    # Crear la Base de datos, clase chroma, con todos los fragmentos combinados
    vectordb = Chroma.from_documents(todos_los_fragmentos,
                                     OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key),
                                     persist_directory="./Chroma_db")

    return vectordb


# Título de la aplicación en Streamlit
st.title("Reglamento del Congreso de la República")
st.write("Este agente responde preguntas relacionadas con el Reglamento del Congreso de la República")

# Widget para cargar múltiples documentos PDF
archivos_subidos = st.file_uploader("Subir documentos en PDF", accept_multiple_files=True, type="pdf")

# Crear botón para procesar los archivos subidos
if st.button("Actualizar Embeddings") and archivos_subidos:
    vectordb = actualizar_embeddings(archivos_subidos)
    st.write("Embeddings actualizado correctamente")

# Pregunta del usuario
pregunta = st.text_area("Haz una pregunta sobre el Reglamento del Congreso")

# Inicializar la clave 'pregunta' en session_state si no existe
if "pregunta" not in st.session_state:
    st.session_state["pregunta"] = ""
# Guardar la pregunta en session_state cada vez que cambie
st.session_state["pregunta"] = pregunta

# Configuración del modelo
prompt_template = """Eres un agente de ayuda inteligente especializado en el Reglamento del Congreso de la República del Perú.
                    Responde las preguntas de los usuarios {input} relacionadas con el reglamento del Congreso basándote estrictamente en el {context}.
                    No hagas suposiciones ni proporciones información que no esté incluida en el {context}."""

llm = ChatOpenAI(model="gpt-4o", max_tokens=1024, api_key=openai_api_key)
qa_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

# Botón para enviar la pregunta
if st.button("Enviar"):
    if st.session_state["pregunta"]:  # Usar la pregunta almacenada en session_state
        vectordb = Chroma(persist_directory="./Chroma_db",
                          embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key))
        resultados_similares = vectordb.similarity_search(st.session_state["pregunta"], k=10)

        contexto = ""
        for doc in resultados_similares:
            contexto += doc.page_content

        respuesta = qa_chain.invoke({"input": st.session_state["pregunta"], "context": contexto})
        resultado = respuesta["text"]
        st.write(resultado)
    else:
        st.write("Por favor, ingresa una pregunta antes de enviar.")
