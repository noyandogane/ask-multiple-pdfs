import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css
import time
import os

load_dotenv()
st.set_page_config(page_title="DocumentGPT", page_icon=":books:")
st.write(css, unsafe_allow_html=True)
st.header("DocumentGPT :books:")

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = 60  # seconds

failed_login_attempts = {}

def authenticate(username, password):
    if username in failed_login_attempts:
        last_attempt_time = failed_login_attempts[username]
        elapsed_time = time.time() - last_attempt_time
        if elapsed_time < LOCKOUT_DURATION:
            return False
    
    env_password = os.getenv(f"{username.upper()}_PASSWORD")
    if env_password and password == env_password:
        failed_login_attempts.pop(username, None)
        return True
    
    if username in failed_login_attempts:
        failed_login_attempts[username] += 1
    else:
        failed_login_attempts[username] = 1
    
    if failed_login_attempts[username] >= MAX_LOGIN_ATTEMPTS:
        failed_login_attempts[username] = time.time()
    
    return False

def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.success("Login successful!")
            return True
        else:
            st.error("Invalid username or password.")
    return False

class DocumentHandler:
    def __init__(self, pdf_docs):
        self.pdf_docs = pdf_docs

    def get_pdf_text(self):
        text = ""
        for pdf in self.pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

class TextHandler:
    def __init__(self, text):
        self.text = text

    def get_text_chunks(self):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_text(self.text)

class VectorStoreHandler:
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def get_vectorstore(self):
        embeddings = OpenAIEmbeddings()
        return FAISS.from_texts(texts=self.text_chunks, embedding=embeddings)

class ConversationHandler:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def get_conversation_chain(self):
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        return ConversationalRetrievalChain.from_llm(llm=llm, retriever=self.vectorstore.as_retriever(), memory=memory)

class UserInputHandler:
    @staticmethod
    def handle_user_input(user_question):
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in reversed(list(enumerate(st.session_state.chat_history))):
            if i % 2 == 0:  # Assuming this is the user's message
                msg = "<div style='text-align: right; color: blue; border:1px solid blue; padding: 10px; margin: 10px; border-radius: 10px;'>{}</div>".format(
                    message.content)
                st.markdown(msg, unsafe_allow_html=True)
            else:  # Assuming this is the bot's message
                msg = "<div style='text-align: left; color: green; border:1px solid green; padding: 10px; margin: 10px; border-radius: 10px;'>{}</div>".format(
                    message.content)
                st.markdown(msg, unsafe_allow_html=True)

def main():
    if login():
        user_question = st.text_input("Ask a question about your documents:")

        if user_question:
            with st.spinner("Thinking..."):
                UserInputHandler.handle_user_input(user_question)

        with st.sidebar:
            st.subheader("Your documents")
            pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    doc_handler = DocumentHandler(pdf_docs)
                    raw_text = doc_handler.get_pdf_text()

                    text_handler = TextHandler(raw_text)
                    text_chunks = text_handler.get_text_chunks()

                    vectorstore_handler = VectorStoreHandler(text_chunks)
                    vectorstore = vectorstore_handler.get_vectorstore()

                    conv_handler = ConversationHandler(vectorstore)
                    st.session_state.conversation = conv_handler.get_conversation_chain()

                st.success("Done! You can now ask questions about your documents.")
    else:
        st.warning("Please log in to access the application.")

if __name__ == '__main__':
    main()
