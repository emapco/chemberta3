import boto3
import os
import tempfile


def copy_file_from_s3(s3_path: str) -> str:
    """
    Copy a file from S3 to local disk.

    Parameters
    ----------
    s3_path: str
        S3 path to file.

    Returns
    -------
    local_path: str
        Local path to file.
    """
    s3 = boto3.client("s3")
    filename = os.path.basename(s3_path)
    local_path = os.path.join(tempfile.gettempdir(), filename)
    s3.download_file(
        s3_path.split("/")[2], "/".join(s3_path.split("/")[3:]), local_path
    )
    return local_path


def check_s3_path(s3_path) -> bool:
    """
    Check if a path is an S3 path.

    Parameters
    ----------
    s3_path: str
        Path to check.

    Returns
    -------
    is_s3_path: bool
        True if path is an S3 path, False otherwise.
    """
    if s3_path.startswith("s3://"):
        return True
    else:
        return False
