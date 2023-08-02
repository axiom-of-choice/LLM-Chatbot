import streamlit as st

from dotenv import load_dotenv

load_dotenv()

from settings import logging_config
import logging.config

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)
from config import BACKGROUNDS_DIR, TITLE, LOGO_DIR
from random import randint
from src.utils import set_background, set_logo
from PIL import Image


def upload_background():
    with st.form("background-form", clear_on_submit=True):
        files = st.file_uploader(
            "Choose a file to upload as background",
            key=st.session_state["widget_background_key"],
            accept_multiple_files=False,
            kwargs={"clear_on_submit": True},
            type=["png", "jpg", "jpeg"],
        )
        submitted = st.form_submit_button("submit")
        if submitted:
            logger.debug(f"Setting background")
            if files is not None:
                # To read file as bytes:
                uploaded_file = files.read()
                with open(BACKGROUNDS_DIR, "wb") as f:
                    logger.debug(f"Writing background")
                    f.write(uploaded_file)
                    logger.debug(f"Writed background")
            st.write("Background set!")
        st.session_state["widget_key"] = str(randint(1000, 100000000))
        set_background(BACKGROUNDS_DIR)


def add_logo():
    with st.form("logo-form", clear_on_submit=True):
        files = st.file_uploader(
            "Choose a file to upload as Logo",
            key=st.session_state["widget_logo_key"],
            accept_multiple_files=False,
            kwargs={"clear_on_submit": True},
            type=["png", "jpg", "jpeg"],
        )
        submitted = st.form_submit_button("submit")
        if submitted:
            logger.debug(f"Setting logo")
            if files is not None:
                # To read file as bytes:
                uploaded_file = files.read()
                with open(LOGO_DIR, "wb") as f:
                    logger.debug(f"Writing logo")
                    f.write(uploaded_file)
                with open(LOGO_DIR, "rb") as f:
                    logger.debug(f"Reading logo to resize")
                    image = Image.open(f)
                    image.resize((500, 500)).save(LOGO_DIR)
                    logger.debug(f"Writed logo")


def change_title():
    with st.form("title-form", clear_on_submit=True):
        title = st.text_input("Title", key=st.session_state["widget_title_key"])
        submitted = st.form_submit_button("submit")
        if submitted:
            logger.debug(f"Setting title")
            if title is not None:
                st.session_state["widget_title_key"] = title
                TITLE = title
                logger.debug(f"Setted title")
                st.write("Title set!")


def config_page(user):
    if "widget_background_key" not in st.session_state:
        st.session_state["widget_background_key"] = randint(0, 1000000)
    else:
        st.session_state["widget_background_key"] = st.session_state["widget_background_key"]
    if "widget_logo_key" not in st.session_state:
        st.session_state["widget_logo_key"] = randint(0, 1000000)
    else:
        st.session_state["widget_logo_key"] = st.session_state["widget_logo_key"]
    if "widget_title_key" not in st.session_state:
        st.session_state["widget_title_key"] = randint(0, 1000000)
    else:
        st.session_state["widget_title_key"] = st.session_state["widget_title_key"]
    if user["su"] == 1:
        st.subheader("Welcome to the admin console.")
        upload_background()
        add_logo()
        change_title()
    else:
        st.write("You are not authorized to view this page.")
