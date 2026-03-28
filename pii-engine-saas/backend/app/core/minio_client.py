"""
MinIO (S3-compatible) object storage client for the PII Engine SaaS platform.

Handles bucket creation, file upload/download, deletion,
and pre-signed URL generation.
"""

from __future__ import annotations

import io
import logging
from datetime import timedelta
from typing import Optional

from minio import Minio
from minio.error import S3Error

from app.config import get_settings

logger = logging.getLogger(__name__)

# Buckets required by the platform
REQUIRED_BUCKETS = ("ml-models", "uploads", "exports", "reports")


class MinIOClient:
    """Manages the MinIO S3-compatible storage client."""

    def __init__(self) -> None:
        self._client: Optional[Minio] = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Initialise the MinIO client and ensure required buckets exist.

        Note: The MinIO Python SDK is synchronous, so this method is not async.
        """
        settings = get_settings()
        logger.info("Connecting to MinIO at %s ...", settings.MINIO_ENDPOINT)

        self._client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )

        self.create_buckets()
        logger.info("MinIO client initialised with buckets: %s", ", ".join(REQUIRED_BUCKETS))

    def get_client(self) -> Minio:
        """Return the active MinIO client.

        Raises:
            RuntimeError: If the client has not been connected yet.
        """
        if self._client is None:
            raise RuntimeError("MinIO not connected. Call connect() first.")
        return self._client

    # ── Bucket Management ────────────────────────────────────────────────

    def create_buckets(self) -> None:
        """Create all required buckets if they do not already exist."""
        client = self.get_client()
        for bucket_name in REQUIRED_BUCKETS:
            try:
                if not client.bucket_exists(bucket_name):
                    client.make_bucket(bucket_name)
                    logger.info("Created MinIO bucket: %s", bucket_name)
                else:
                    logger.debug("MinIO bucket already exists: %s", bucket_name)
            except S3Error as exc:
                logger.error("Failed to create bucket '%s': %s", bucket_name, exc)
                raise

    # ── File Operations ──────────────────────────────────────────────────

    def upload_file(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict[str, str]] = None,
    ) -> str:
        """Upload a file (bytes) to a MinIO bucket.

        Args:
            bucket: Target bucket name.
            object_name: Object key / path within the bucket.
            data: Raw file bytes.
            content_type: MIME type of the object.
            metadata: Optional user-defined metadata headers.

        Returns:
            The object name that was written.
        """
        client = self.get_client()
        stream = io.BytesIO(data)
        client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=stream,
            length=len(data),
            content_type=content_type,
            metadata=metadata,
        )
        logger.info("Uploaded %s/%s (%d bytes).", bucket, object_name, len(data))
        return object_name

    def download_file(self, bucket: str, object_name: str) -> bytes:
        """Download a file from a MinIO bucket.

        Args:
            bucket: Source bucket name.
            object_name: Object key / path within the bucket.

        Returns:
            Raw file bytes.

        Raises:
            S3Error: If the object does not exist or is inaccessible.
        """
        client = self.get_client()
        response = None
        try:
            response = client.get_object(bucket_name=bucket, object_name=object_name)
            data = response.read()
            logger.info("Downloaded %s/%s (%d bytes).", bucket, object_name, len(data))
            return data
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def delete_file(self, bucket: str, object_name: str) -> None:
        """Delete a file from a MinIO bucket.

        Args:
            bucket: Bucket name.
            object_name: Object key / path to delete.
        """
        client = self.get_client()
        client.remove_object(bucket_name=bucket, object_name=object_name)
        logger.info("Deleted %s/%s.", bucket, object_name)

    def get_presigned_url(
        self,
        bucket: str,
        object_name: str,
        *,
        expires: timedelta = timedelta(hours=1),
        method: str = "GET",
    ) -> str:
        """Generate a pre-signed URL for temporary access to an object.

        Args:
            bucket: Bucket name.
            object_name: Object key / path.
            expires: URL expiry duration (default 1 hour).
            method: HTTP method (``GET`` for download, ``PUT`` for upload).

        Returns:
            Pre-signed URL string.
        """
        client = self.get_client()

        if method.upper() == "PUT":
            url = client.presigned_put_object(
                bucket_name=bucket,
                object_name=object_name,
                expires=expires,
            )
        else:
            url = client.presigned_get_object(
                bucket_name=bucket,
                object_name=object_name,
                expires=expires,
            )

        logger.debug("Generated pre-signed %s URL for %s/%s.", method, bucket, object_name)
        return url


# Module-level singleton
minio_client = MinIOClient()
