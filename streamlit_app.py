"""
# My first app
Here's our first attempt at using data to create a table:
"""
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(page_title="HugChat - An LLM-powered Streamlit app")

########### Vector DB AND MODEL ############
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from config.config import BASE_DIR, DATA_DIR, EMBEDDING_MODEL_NAME, PINECONE_INDEX_NAME
from src.data.parser import connect_index
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


# Save it into pinecone
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT", "us-west4-gcp-free")
OPENAI_API_KEY = os.environ.get("OPEN_AI_KEY")

load_dotenv()


index = connect_index(PINECONE_INDEX_NAME,PINECONE_API_KEY, PINECONE_ENV)
embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL_NAME)
vectorstore = Pinecone(index, embeddings.embed_query, 'text')



# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)



qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
########### Vector DB AND MODEL ############


if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm HugChat, How may I help you?"]
else:
    st.session_state['generated'] = st.session_state['generated']

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']
else:
    st.session_state['past'] = st.session_state['past']




with st.sidebar:
    st.title('🤗💬 HugChat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](<https://streamlit.io/>)
    - [HugChat](<https://github.com/Soulter/hugging-chat-api>)
    - [Open AI Model]) LLM model
    
    💡 Note: No API key required!
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Data Professor](<https://youtube.com/dataprofessor>)')
    
    
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

## Applying the user input box
with input_container:
    user_input = get_text()
    
# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    chatbot = qa_with_sources
    response = chatbot(prompt)
    return response['answer'] + ' (Source: ' + response['sources'] + ')'

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state['generated'][i], key=str(i))