from . import s3, local


DATA_SOURCE = {"S3": s3.interface, "Local": local.interface}
