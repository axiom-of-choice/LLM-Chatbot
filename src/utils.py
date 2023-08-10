import pinecone
import logging
from config import *
import streamlit as st

logger = logging.getLogger(__name__)


@st.experimental_singleton
def connect_index(index_name: str, API_KEY: str = PINECONE_API_KEY, ENV: str = PINECONE_ENV) -> pinecone.Index:
    pinecone.init(api_key=API_KEY, environment=ENV)
    index = pinecone.Index(index_name)
    logger.info(f"Connected to Pinecone index {index_name}")
    return index


@st.experimental_singleton
def init_pinecone_index(api_key: str = PINECONE_API_KEY, environment: str = PINECONE_ENV) -> None:
    pinecone.init(api_key=api_key, environment=environment)
    logger.info("Pinecone initialized")
    return None


import base64


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = (
        """
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    """
        % bin_str
    )
    st.markdown(page_bg_img, unsafe_allow_html=True)


def set_background_remote(url):
    page_bg_img = (
        """
    <style>
    .stApp {
    background-image: url("%s");
    background-size: cover;
    }
    </style>
    """
        % url
    )
    st.markdown(page_bg_img, unsafe_allow_html=True)


def set_logo(file):
    with open(file, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    st.sidebar.markdown(
        f"""
        <div style="display:table;margin-top:-20%;margin-left:20%;">
            <img src="data:image/png;base64,{data}" width="100" height="150">
        </div>
        """,
        unsafe_allow_html=True,
    )


def set_logo_remote(url):
    st.sidebar.markdown(
        f"""
        <div style="display:table;margin-top:-20%;margin-left:20%;">
            <img src="{url}" width="100" height="150">
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    connect_index("stab-test")
