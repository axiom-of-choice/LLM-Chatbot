"""
# My first app
Here's our first attempt at using data to create a table:
"""
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(page_title="AS - An LLM-powered Streamlit app")
from streamlit_extras.app_logo import add_logo

########### Vector DB AND MODEL ############
from dotenv import load_dotenv
from config import *


import logging
from settings import logging_config
import logging.config

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

load_dotenv()


##TITLE background and logo
import toml

client_config = toml.load(os.path.join(BASE_DIR, "client_config.toml"))
TITLE = "AS - An LLM-powered Streamlit app"

##PAGEs
from pages import AVAILABLE_PAGES

main_page = AVAILABLE_PAGES["Main Page"]


with st.sidebar:
    st.title("Generative QA LLM-powered Chatbot")
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
    st.write("Made with ❤️ by [Isaac Hernandez Garcia](https://www.linkedin.com/in/isaac-hernandez-garcia-9905/)")
    # selected_page = st.sidebar.selectbox("Select a page", AVAILABLE_PAGES.keys())


st.title(TITLE)
#############
main_page({})
