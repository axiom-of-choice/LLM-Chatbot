"""
# My first app
Here's our first attempt at using data to create a table:
"""
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(page_title="InsightFinder - An LLM-powered Streamlit app")

########### Vector DB AND MODEL ############
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from config.config import *
from src.data.cohere_parser import parse
from src.utils import connect_index
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import CohereEmbeddings
import glob
import traceback
from random import randint


import logging
from settings import logging_config
import logging.config
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

load_dotenv()


############# AUTH #############
import streamlit as st

from autentication import streamlit_debug
streamlit_debug.set(flag=False, wait_for_client=True, host='localhost', port=8765)

from autentication import env
env.verify()

from autentication.authlib.auth import auth, authenticated, requires_auth
from autentication.authlib.common import trace_activity

user = auth(sidebar=True, show_msgs=True)
with st.sidebar:
        st.title('🤗💬 InsightFinder App')
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](<https://streamlit.io/>)
        - [Pinecone Vector Database](<https://www.pinecone.io/>)
        - [Cohere Multi lingual Embeddings](<https://cohere.ai/>)
        - Open AI gpt-3.5 turbo for Gen QA
        💡 Note: No API key required!
        ''')
        add_vertical_space(5)
        st.write('Made with ❤️ by [Isaac Hernandez Garcia](https://www.linkedin.com/in/isaac-hernandez-garcia-9905/)')

st.title('Insight Finder App')
if not authenticated():
    st.warning(f'Please log in to access the app')
#############
else:
    st.success(f'`{user}` is authenticated')
    index = connect_index(PINECONE_INDEX_NAME,PINECONE_API_KEY, PINECONE_ENV)
    embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model=COHERE_MODEL_NAME)
    vectorstore = Pinecone(index, embeddings.embed_query, 'text')
    temp_data = os.path.join(DATA_DIR, 'tmp/')



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


    ###SESSION STATE###
    if 'widget_key' not in st.session_state:
        st.session_state['widget_key'] = randint(0, 1000000)
    else:
        st.session_state['widget_key'] = st.session_state['widget_key']

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["I'm a ChatBot that only can answers questions, ask me anything!"]
    else:
        st.session_state['generated'] = st.session_state['generated']

    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hi!']
    else:
        st.session_state['past'] = st.session_state['past']

    with st.form("my-form", clear_on_submit=True):
        files = st.file_uploader("Choose a file to upload and ask a question about it:", key=st.session_state['widget_key'], accept_multiple_files=True, kwargs={'clear_on_submit': True})
        if len(files) >0:
            for uploaded_file in files:
                if uploaded_file is not None:
            # To read file as bytes:
                    bytes_data = uploaded_file.read()
                    # st.write("filename:", uploaded_file.name)
                    logger.info(f"Writing: {uploaded_file.name}")
                    with open(f"{temp_data}/{uploaded_file.name}", 'wb') as f:
                        f.write(bytes_data)
        logger.info(f"Parsing files {files}")
        submitted = st.form_submit_button("submit")
        if submitted:
            st.session_state['widget_key'] = str(randint(1000, 100000000))
        try:
            parse(temp_data, output_filepath= None, index_name= PINECONE_INDEX_NAME, embeddings_model_name= COHERE_MODEL_NAME, glob=GLOB)
        except:
            logger.error("Error parsing files")
            logger.error(traceback.format_exc())
        files = []
        files = glob.glob(f'{temp_data}/*')
        for f in files:
            os.remove(f)
        logger.info(f"Files removed from {temp_data}")



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
