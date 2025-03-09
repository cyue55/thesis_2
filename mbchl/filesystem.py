import io
import os
import posixpath
import time
import warnings

import boto3
import botocore
import requests
import sshfs
from botocore.config import Config


def init_filesystem(base_url):
    """Initialize a filesystem object.

    ``base_url`` must be a local path or an URL with the form ``"protocol://..."``.
    Supported protocols are HTTPS, S3 and SSH.

    - If a local path, methods from the filesystem instance such as :meth:`open` will
      be relative to ``base_url``. For example, to use paths relative to the current
      working directory, use ``init_filesystem("")``.
    - HTTPS: use the form ``"https://example.com/..."``. Subsequent calls to
      methods with argument ``path`` will request ``"https://example.com/.../path"``.
    - S3: use the form ``"s3://bucket/..."``. Subsequent calls to methods with argument
      ``path`` will request the object at ``".../path"`` in the S3 bucket. The
      environment variables ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY`` and
      ``AWS_ENDPOINT_URL`` must be set.
    - SSH: use the form ``"ssh://username@host:..."``. Subsequent calls to methods with
      argument ``path`` will request ``.../path`` on the remote host. Requires an SSH
      client and a public key installed on the remote host.

    """
    if base_url.startswith("ssh://"):
        username_and_host, base_path = base_url[6:].split(":", 1)
        username, host = username_and_host.split("@", 1)
        fs = SSHFileSystem(host, username, base_path)
    elif base_url.startswith(("http://", "https://")):
        fs = HTTPFileSystem(base_url)
    elif base_url.startswith("s3://"):
        bucket_name, base_path = base_url[5:].split("/", 1)
        fs = S3FileSystem(bucket_name, base_path)
    else:
        fs = LocalFileSystem(base_url)
    return fs


class LocalFileSystem:
    """Local filesystem object."""

    def __init__(self, base_path):
        self.base_path = base_path

    def open(self, path, mode="rb", offset=None):
        """Open a file and seek to offset."""
        full_path = self.fullpath(path)
        f = open(full_path, mode=mode)
        if offset is not None:
            f.seek(offset)
        return f

    def exists(self, path):
        """Check if file exists."""
        full_path = self.fullpath(path)
        return os.path.exists(full_path)

    def makedirs(self, path):
        """Create directory."""
        full_path = self.fullpath(path)
        os.makedirs(full_path, exist_ok=True)

    def put(self, fileobj, path):
        """Put data from file object to file."""
        full_path = self.fullpath(path)
        data = fileobj.read()
        with open(full_path, "wb") as f:
            f.write(data)

    def get(self, path):
        """Get data from file."""
        full_path = self.fullpath(path)
        buffer = io.BytesIO()
        with open(full_path, "rb") as f:
            data = f.read()
        buffer.write(data)
        buffer.seek(0)
        return buffer

    def join(self, *paths):
        """Join paths. Equivalent to :func:`os.path.join`."""
        return os.path.join(*paths)

    def dirname(self, path):
        """Return directory name. Equivalent to :func:`os.path.dirname`."""
        return os.path.dirname(path)

    def remove(self, path):
        """Remove file."""
        full_path = self.fullpath(path)
        os.remove(full_path)

    def fullpath(self, path):
        """Return full path."""
        return self.join(self.base_path, path)

    def size(self, path):
        """Return file size."""
        full_path = self.fullpath(path)
        return os.path.getsize(full_path)


class SSHFileSystem(sshfs.SSHFileSystem):
    """SSH filesystem object."""

    def __init__(self, host, username, base_path):
        super().__init__(host, username=username)
        self.base_path = base_path

    def open(self, path, mode="rb", offset=None):
        """Open a file and seek to offset."""
        full_path = posixpath.join(self.base_path, path)
        f = super().open(full_path, mode=mode)
        if offset is not None:
            f.seek(offset)
        return f

    def size(self, path):
        """Return file size."""
        full_path = posixpath.join(self.base_path, path)
        return super().size(full_path)


class HTTPFileSystem:
    """HTTP filesystem object."""

    def __init__(self, base_url):
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid base url: {base_url}")
        self.base_url = base_url
        self._file_path = None
        self._file_size = None

    def open(self, path, mode="rb", offset=None):
        """Open a file and seek to offset."""
        if mode != "rb":
            raise ValueError(f"{self.__class__.__name__} only supports 'rb' mode")
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        file = HTTPFile(url, offset=offset)
        self._file_path = path
        self._file_size = file.size
        return file

    def size(self, path):
        """Return file size."""
        if path == self._file_path:
            return self._file_size
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        response = requests.head(url)
        response.raise_for_status()
        return int(response.headers["Content-Length"])


class HTTPFile:
    """HTTP file object."""

    def __init__(self, url, offset=None):
        self.url = url
        self.response = self._get_response(offset)
        self.position = 0 if offset is None else offset

    def _get_response(self, offset=None):
        headers = {} if offset is None else {"Range": f"bytes={offset}-"}
        response = requests.get(self.url, headers=headers, stream=True)
        response.raise_for_status()
        return response

    def seek(self, offset):
        """Seek to offset."""
        self.response.close()
        self.response = self._get_response(offset)
        self.position = offset
        return offset

    def tell(self):
        """Return current position."""
        return self.position

    def read(self, size=-1):
        """Read data from file."""
        data = self.response.raw.read(size)
        self.position += len(data)
        return data

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.response.close()

    @property
    def size(self):
        """Return file size."""
        byte_range = self.response.headers.get("Content-Range")
        if byte_range is None:
            return int(self.response.headers["Content-Length"])
        return int(byte_range.split("/")[-1])


class S3FileSystem:
    """S3 filesystem object."""

    def __init__(self, bucket_name, base_path):
        self.bucket_name = bucket_name
        self.base_path = base_path
        self.s3 = boto3.client(
            "s3", config=Config(retries={"max_attempts": 10, "mode": "standard"})
        )
        self._file_path = None
        self._file_size = None

    def open(self, path, mode="rb", offset=None):
        """Open a file and seek to offset."""
        if mode != "rb":
            raise ValueError(f"{self.__class__.__name__} only supports 'rb' mode")
        full_path = self.fullpath(path)
        file = S3File(self.s3, self.bucket_name, full_path, offset=offset)
        self._file_path = path
        self._file_size = file.size
        return file

    def size(self, path):
        """Return file size."""
        if path == self._file_path:
            return self._file_size
        full_path = self.fullpath(path)
        response = self.s3.head_object(Bucket=self.bucket_name, Key=full_path)
        return int(response["ContentLength"])

    def exists(self, path, max_attempts=3):
        """Check if file exists."""
        err = None
        full_path = self.fullpath(path)
        for _ in range(max_attempts):
            try:
                self.s3.head_object(Bucket=self.bucket_name, Key=full_path)
                return True
            except botocore.exceptions.ClientError as e:
                if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                    return False
                err = e
            _wait_before_retry()
        if err is not None:
            raise S3FileSystemError(
                f"Failed to check existence of {self.bucket_name}/{path}"
            ) from err

    def makedirs(self, path):
        """Do nothing as empty directories do not exist in S3."""
        pass

    def put(self, fileobj, path):
        """Put data from file object to file."""
        full_path = self.fullpath(path)
        try:
            self.s3.upload_fileobj(fileobj, self.bucket_name, full_path)
        except botocore.exceptions.ClientError as e:
            raise S3FileSystemError(
                f"Failed to upload {self.bucket_name}/{path}"
            ) from e

    def get(self, path, max_attempts=3):
        """Get data from file."""
        full_path = self.fullpath(path)
        buffer = io.BytesIO()
        err = None
        for _ in range(max_attempts):
            try:
                self.s3.download_fileobj(self.bucket_name, full_path, buffer)
                buffer.seek(0)
                return buffer
            except botocore.exceptions.ClientError as e:
                err = e
            _wait_before_retry()
        if err is not None:
            raise S3FileSystemError(
                f"Failed to download {self.bucket_name}/{path}"
            ) from err
        buffer.seek(0)
        return buffer

    def join(self, *paths):
        """Join paths. Equivalent to :func:`posixpath.join`."""
        return posixpath.join(*paths)

    def dirname(self, path):
        """Return directory name. Equivalent to :func:`posixpath.dirname`."""
        return posixpath.dirname(path)

    def remove(self, path):
        """Remove file."""
        full_path = self.fullpath(path)
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=full_path)
        except botocore.exceptions.ClientError as e:
            raise S3FileSystemError(
                f"Failed to delete {self.bucket_name}/{path}"
            ) from e

    def fullpath(self, path):
        """Return full path."""
        return self.join(self.base_path, path)


class S3File:
    """S3 file object."""

    def __init__(self, s3, bucket_name, path, offset=None):
        self.s3 = s3
        self.bucket_name = bucket_name
        self.path = path
        self.response = self._get_response(offset)
        self.position = 0 if offset is None else offset

    def _get_response(self, offset=None, max_attempts=3):
        range_header = f"bytes={offset}-" if offset is not None else None
        err = None
        for _ in range(max_attempts):
            try:
                return self.s3.get_object(
                    Bucket=self.bucket_name, Key=self.path, Range=range_header
                )
            except botocore.exceptions.ClientError as e:
                if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                    raise FileNotFoundError(f"{self.bucket_name}/{self.path}") from e
                err = e
            _wait_before_retry()
        if err is not None:
            raise S3FileSystemError(
                f"Failed to get {self.bucket_name}/{self.path}"
            ) from err

    def seek(self, offset):
        """Seek to offset."""
        self.response["Body"].close()
        self.response = self._get_response(offset)
        self.position = offset
        return offset

    def tell(self):
        """Return current position."""
        return self.position

    def read(self, size=-1):
        """Read data from file."""
        try:
            data = self.response["Body"].read(size)
        except botocore.exceptions.ResponseStreamingError:
            warnings.warn("Connection to S3 bucket lost. Trying to reconnect.")
            self.response["Body"].close()
            self.response = self._get_response(self.position)
            data = self.response["Body"].read(size)
        self.position += len(data)
        return data

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.response["Body"].close()

    @property
    def size(self):
        """Return file size."""
        byte_range = self.response.get("ContentRange")
        if byte_range is None:
            return int(self.response["ContentLength"])
        return int(byte_range.split("/")[-1])


class S3FileSystemError(Exception):
    """S3 filesystem error."""


def _wait_before_retry():
    """Wait before retrying S3 operation."""
    time.sleep(3.0)
