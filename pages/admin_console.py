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
import os
import pandas as pd
import json
import boto3

from dotenv import load_dotenv

load_dotenv()

from connectors import DATA_SOURCE


def admin_console(user):
    if user["su"] == 1:
        st.write("Welcome to the admin console.")
        selected_datasource = st.selectbox("Select a Datasoruce", DATA_SOURCE.keys())
        obj = DATA_SOURCE[selected_datasource]
        obj().interface()
    else:
        st.write("You are not authorized to view this page.")
