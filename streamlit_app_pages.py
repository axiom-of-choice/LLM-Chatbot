import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(page_title="InsightFinder - An LLM-powered Streamlit app")
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


############# AUTH #############
import streamlit as st

from autentication import streamlit_debug

streamlit_debug.set(flag=False, wait_for_client=True, host="localhost", port=8765)

from autentication import env

env.verify()

from autentication.authlib.auth import auth, authenticated, requires_auth
from autentication.authlib.common import trace_activity

##TITLE background and logo
import toml
from src.utils import set_background, set_logo, set_background_remote, set_logo_remote

client_config = toml.load(os.path.join(BASE_DIR, "client_config.toml"))
TITLE = client_config["branding"]["title"]
# set_background_remote(BACKGROUNDS_DIR)
set_logo_remote(LOGO_DIR)

##PAGES
from pages import AVAILABLE_PAGES


user = auth(sidebar=True, show_msgs=True)
with st.sidebar:
    st.title(TITLE)
    st.markdown(
        # """
        # ## About
        # This app is an LLM-powered chatbot built using:
        # - [Streamlit](<https://streamlit.io/>)
        # - [Pinecone Vector Database](<https://www.pinecone.io/>)
        # - [Cohere Multi lingual Embeddings](<https://cohere.ai/>)
        # - Open AI gpt-3.5 turbo for Gen QA
        # 💡 Note: No API key required!
        # """
        """
    ## About
    This app is an LLM-powered chatbot that answers question over a knowledge base.
    """
    )
    add_vertical_space(5)
    st.write("Made with ❤️ by [Isaac Hernandez Garcia](https://www.linkedin.com/in/isaac-hernandez-garcia-9905/)")
    selected_page = st.sidebar.selectbox("Select a page", AVAILABLE_PAGES.keys())


st.title(TITLE)
if not authenticated():
    st.warning(f"Please log in to access the app")
#############
else:
    AVAILABLE_PAGES[selected_page](user)
