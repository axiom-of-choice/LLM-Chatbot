from . import s3, local


DATA_SOURCE = {"S3": s3.S3Connector, "Local": local.LocalConnector}
