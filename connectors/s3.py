import boto3
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
import streamlit as st


S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
S3_LOCATION = os.environ.get("S3_LOCATION")


class S3Connector:
    def __init__(
        self, location=S3_LOCATION, access_key=S3_ACCESS_KEY, secret_key=S3_SECRET_KEY
    ):
        self.location = location
        self.access_key = access_key
        self.secret_key = secret_key
        self._client = boto3.client(
            "s3",
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

    def display_objets_in_bucket(self, bucket_name):
        response = self._client.list_objects(
            Bucket=bucket_name,
        )
        response = response["Contents"]
        response = pd.DataFrame(response)
        return response

    def get_object_url(self, bucket_name, key):
        return f'<a target="_blank" href="https://{bucket_name}.s3.amazonaws.com/{key}">{key}</a>'
        # f'<a target="_blank" href="https://{selected_bucket}.s3.amazonaws.com/{row["Key"]}">{row["Key"]}</a>'

    def interface(self):
        """Create an interface for navigating between buckets and files in S3

        Args:
            user (_type_): User for permissions boundaries
        """

        buckets = {
            bucket["Name"]: bucket["CreationDate"]
            for bucket in self._client.list_buckets()["Buckets"]
        }
        # response = pd.DataFrame(response)
        # st.write(buckets)
        selected_bucket = st.selectbox(
            "Select a Bucket to list objects", buckets.keys()
        )
        try:
            df = self.display_objets_in_bucket(selected_bucket)
            df["URL"] = df.apply(
                lambda row: self.get_object_url(selected_bucket, row["Key"]), axis=1
            )
            df["Size"] = df["Size"] / 1000000  # Convert to MB
            df = df[["Key", "LastModified", "Size", "URL"]]
            df.columns = ["Key", "LastModified", "Size in MB", "URL"]
            st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        except Exception as e:
            st.write("No objects in this bucket")
            st.write(e)

    def upload_file(self, obj, bucket_name, file_name):
        """Uploads a file to an S3 bucket
        Args:
            coll_name (str): Collection to be uploaded to (Folder in S3)
            file_name (str): File to be uploaded
        Returns:
            str: Response from S3
        """
        try:
            self._client.put_object(Body=obj, Bucket=bucket_name, Key=file_name)
            return "File uploaded successfully"
        except Exception as e:
            return e

    def download_file(self, bucket_name, file_name):
        """Download a file from S3
        Args:
            coll_name (str): Collection to be uploaded to (Folder in S3)
            file_name (str): File to be uploaded
        Returns:
            str: Response from S3
        """
        try:
            self._client.download_file(bucket_name, file_name, file_name)
            return "File downloaded successfully"
        except Exception as e:
            return e

    def upload_multiple_files(self, dir, bucket_name):
        """Uploads multiple files to an S3 bucket
        Args:
            coll_name (str): Collection to be uploaded to (Folder in S3)
            dir (str): Directory to be uploaded
        Returns:
            str: Response from S3
        """
        try:
            for file in os.listdir(dir):
                if not file.startswith("."):
                    self._client.upload_file(os.path.join(dir, file), bucket_name, file)
            return "Files uploaded successfully"
        except Exception as e:
            return e


if __name__ == "__main__":
    s3 = S3Connector()
    print(s3.display_objets_in_bucket("insightfinder-test"))
