# streamlit_app.py

# import streamlit as st
# from st_files_connection import FilesConnection
#
# # Create connection object and retrieve file contents.
# # Specify input format is a csv and to cache the result for 600 seconds.
# conn = st.experimental_connection('s3', type=FilesConnection)
# df = conn.read("abinveb-bucket/common.txt", input_format="text", ttl=600)
#
# # Print results.
#
# st.write(df)

import streamlit as st

from dotenv import load_dotenv

load_dotenv()

from connectors import DATA_SOURCE

from settings import logging_config
import logging.config

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)
from config import BACKGROUNDS_DIR, TITLE
from random import randint


def interface(user):
    if "widget_background_key" not in st.session_state:
        st.session_state["widget_background_key"] = randint(0, 1000000)
    else:
        st.session_state["widget_background_key"] = st.session_state["widget_background_key"]
    if "widget_logo_key" not in st.session_state:
        st.session_state["widget_logo_key"] = randint(0, 1000000)
    else:
        st.session_state["widget_logo_key"] = st.session_state["widget_logo_key"]
    if user["su"] == 1:
        st.subheader("Welcome to the admin console.")
        selected_datasource = st.selectbox("Select a Datasoruce", DATA_SOURCE.keys())
        obj = DATA_SOURCE[selected_datasource]
        obj().interface()
    else:
        st.write("You are not authorized to view this page.")
