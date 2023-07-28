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

# from langchain.embeddings import HuggingFaceEmbeddings
# import pinecone
# from langchain.vectorstores import Pinecone
# from langchain.prompts import PromptTemplate
# from config.config import *
# from src.data.cohere_parser import parse
# from src.utils import connect_index
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import CohereEmbeddings
# import glob
# import traceback
# from random import randint


import logging
from settings import logging_config
import logging.config

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

load_dotenv()


############# AUTH #############
import streamlit as st

from autentication import streamlit_debug

streamlit_debug.set(flag=False, wait_for_client=True, host="localhost", port=8765)

from autentication import env

env.verify()

from autentication.authlib.auth import auth, authenticated, requires_auth
from autentication.authlib.common import trace_activity

##PAGES
from pages.main_page import main_page
from pages.admin_console import admin_console


page_names_to_funcs = {"Main Page": main_page, "Admin console": admin_console}

user = auth(sidebar=True, show_msgs=True)
with st.sidebar:
    st.title("🤗💬 InsightFinder App")
    st.markdown(
        """
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](<https://streamlit.io/>)
        - [Pinecone Vector Database](<https://www.pinecone.io/>)
        - [Cohere Multi lingual Embeddings](<https://cohere.ai/>)
        - Open AI gpt-3.5 turbo for Gen QA
        💡 Note: No API key required!
        """
    )
    add_vertical_space(5)
    st.write(
        "Made with ❤️ by [Isaac Hernandez Garcia](https://www.linkedin.com/in/isaac-hernandez-garcia-9905/)"
    )
    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())


st.title("Insight Finder App")
if not authenticated():
    st.warning(f"Please log in to access the app")
#############
else:
    page_names_to_funcs[selected_page](user)
