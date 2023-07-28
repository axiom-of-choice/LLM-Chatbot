import boto3
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
import streamlit as st


S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")


def display_objets_in_bucket(client, bucket_name):
    response = client.list_objects(
        Bucket=bucket_name,
    )
    response = response["Contents"]
    response = pd.DataFrame(response)
    return response


def interface(user):
    """Create an interface for navigating between buckets and files in S3

    Args:
        user (_type_): User for permissions boundaries
    """
    client = boto3.client(
        "s3", aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY
    )
    buckets = {
        bucket["Name"]: bucket["CreationDate"]
        for bucket in client.list_buckets()["Buckets"]
    }
    # response = pd.DataFrame(response)
    # st.write(buckets)
    selected_bucket = st.selectbox("Select a Bucket to list objects", buckets.keys())

    try:
        df = display_objets_in_bucket(client, selected_bucket)
        st.write(df)
    except:
        st.write("No objects in this bucket")


def upload_file(obj, bucket_name, file_name):
    """Uploads a file to an S3 bucket
    Args:
        coll_name (str): Collection to be uploaded to (Folder in S3)
        file_name (str): File to be uploaded
    Returns:
        str: Response from S3
    """
    try:
        boto3.client.put_object(Body=object, Bucket=bucket_name, Key=file_name)
        return "File uploaded successfully"
    except Exception as e:
        return e


if __name__ == "__main__":
    pass
