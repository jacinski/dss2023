from dotenv import load_dotenv
load_dotenv()

"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

print("Loading Chroma DB")
persist_directory = 'db'
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
llm = OpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# From here down is all the StreamLit UI.
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def get_text():
    input_text = st.text_input("Question", "What is Langchain?", key="input")
    return input_text


user_input = get_text()

if user_input:
    with st.spinner('In progress..'):
        output = qa(user_input)['result']

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)


if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

