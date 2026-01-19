"""Online Feature Store - Redis-based real-time feature serving."""

from ml_service.feature_store.online.redis_store import OnlineStore, OnlineStoreConfig

__all__ = [
    "OnlineStore",
    "OnlineStoreConfig",
]
