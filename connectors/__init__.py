from . import s3, local
from .s3 import S3Connector
from .local import LocalConnector


DATA_SOURCE = {"S3": s3.S3Connector, "Local": local.LocalConnector}
