"""Offline Feature Store - Parquet + DuckDB."""

from ml_service.feature_store.offline.reader import OfflineStoreReader
from ml_service.feature_store.offline.writer import OfflineStoreWriter

__all__ = [
    "OfflineStoreReader",
    "OfflineStoreWriter",
]
