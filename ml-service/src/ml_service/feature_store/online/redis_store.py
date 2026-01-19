"""Redis-based Online Feature Store for real-time serving."""

from datetime import UTC, datetime
from typing import Any

import redis.asyncio as redis
from pydantic import BaseModel

from ml_service.core.config import get_settings
from ml_service.core.exceptions import FeatureStoreConnectionError


class OnlineStoreConfig(BaseModel):
    """Configuration for Online Store."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    ssl: bool = False
    key_prefix: str = "feature"
    default_ttl_seconds: int | None = None  # Uses settings if None


def get_ttl_policies() -> dict[str, int]:
    """Get TTL policies from settings.

    Returns:
        Dictionary mapping feature group names to TTL in seconds.
    """
    settings = get_settings()
    return {
        "text_features": settings.text_features_ttl_seconds,
        "battle_features": settings.battle_features_ttl_seconds,
        "user_features": settings.user_features_ttl_seconds,
    }


class OnlineStore:
    """Redis-based Online Feature Store.

    Provides low-latency feature retrieval for real-time inference.
    Features are stored as Redis hashes with TTL-based expiration.

    Key structure:
        {prefix}:{entity_type}:{entity_id}:{feature_group}:v{version}

    Example:
        >>> store = OnlineStore(OnlineStoreConfig())
        >>> await store.connect()
        >>> await store.set_features(
        ...     entity_type="text",
        ...     entity_id="123",
        ...     feature_group="text_features",
        ...     features={"text_length": 100, "word_count": 20},
        ... )
        >>> features = await store.get_features(
        ...     entity_type="text",
        ...     entity_id="123",
        ...     feature_group="text_features",
        ... )
    """

    def __init__(self, config: OnlineStoreConfig | None = None) -> None:
        """Initialize the online store.

        Args:
            config: Configuration for Redis connection.
        """
        self.config = config or OnlineStoreConfig()
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        self._client = redis.Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            ssl=self.config.ssl,
            decode_responses=True,
        )
        # Test connection
        await self._client.ping()

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None

    def _get_key(
        self,
        entity_type: str,
        entity_id: str,
        feature_group: str,
        version: int = 1,
    ) -> str:
        """Build Redis key for features.

        Args:
            entity_type: Type of entity (e.g., "text", "battle").
            entity_id: Unique entity identifier.
            feature_group: Name of the feature group.
            version: Feature group version.

        Returns:
            Redis key string.
        """
        return f"{self.config.key_prefix}:{entity_type}:{entity_id}:{feature_group}:v{version}"

    def _get_ttl(self, feature_group: str) -> int:
        """Get TTL for a feature group.

        Args:
            feature_group: Name of the feature group.

        Returns:
            TTL in seconds.
        """
        ttl_policies = get_ttl_policies()
        default_ttl = (
            self.config.default_ttl_seconds
            if self.config.default_ttl_seconds is not None
            else get_settings().default_features_ttl_seconds
        )
        return ttl_policies.get(feature_group, default_ttl)

    async def set_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_group: str,
        features: dict[str, Any],
        version: int = 1,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store features in Redis.

        Args:
            entity_type: Type of entity.
            entity_id: Unique entity identifier.
            feature_group: Name of the feature group.
            features: Dictionary of feature name -> value.
            version: Feature group version.
            ttl_seconds: Optional custom TTL. Uses default if not specified.
        """
        if not self._client:
            raise FeatureStoreConnectionError("Not connected. Call connect() first.")

        key = self._get_key(entity_type, entity_id, feature_group, version)
        ttl = ttl_seconds or self._get_ttl(feature_group)

        # Convert all values to strings for Redis hash
        string_features = {k: str(v) for k, v in features.items()}
        string_features["_updated_at"] = datetime.now(UTC).isoformat()

        # Use pipeline for atomic operation
        async with self._client.pipeline() as pipe:
            await pipe.hset(key, mapping=string_features)
            await pipe.expire(key, ttl)
            await pipe.execute()

    async def get_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_group: str,
        version: int = 1,
    ) -> dict[str, Any] | None:
        """Retrieve features from Redis.

        Args:
            entity_type: Type of entity.
            entity_id: Unique entity identifier.
            feature_group: Name of the feature group.
            version: Feature group version.

        Returns:
            Dictionary of features or None if not found.
        """
        if not self._client:
            raise FeatureStoreConnectionError("Not connected. Call connect() first.")

        key = self._get_key(entity_type, entity_id, feature_group, version)
        result = await self._client.hgetall(key)

        if not result:
            return None

        return self._deserialize_features(result)

    async def get_features_batch(
        self,
        entity_type: str,
        entity_ids: list[str],
        feature_group: str,
        version: int = 1,
    ) -> dict[str, dict[str, Any] | None]:
        """Retrieve features for multiple entities.

        Uses Redis pipeline for efficient batch retrieval.

        Args:
            entity_type: Type of entity.
            entity_ids: List of entity identifiers.
            feature_group: Name of the feature group.
            version: Feature group version.

        Returns:
            Dictionary mapping entity_id -> features (or None if not found).
        """
        if not self._client:
            raise FeatureStoreConnectionError("Not connected. Call connect() first.")

        async with self._client.pipeline() as pipe:
            keys = []
            for entity_id in entity_ids:
                key = self._get_key(entity_type, entity_id, feature_group, version)
                keys.append(key)
                await pipe.hgetall(key)

            results = await pipe.execute()

        return {
            entity_id: self._deserialize_features(result) if result else None
            for entity_id, result in zip(entity_ids, results, strict=True)
        }

    async def delete_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_group: str,
        version: int = 1,
    ) -> bool:
        """Delete features from Redis.

        Args:
            entity_type: Type of entity.
            entity_id: Unique entity identifier.
            feature_group: Name of the feature group.
            version: Feature group version.

        Returns:
            True if key was deleted, False if not found.
        """
        if not self._client:
            raise FeatureStoreConnectionError("Not connected. Call connect() first.")

        key = self._get_key(entity_type, entity_id, feature_group, version)
        deleted = await self._client.delete(key)
        return deleted > 0

    async def exists(
        self,
        entity_type: str,
        entity_id: str,
        feature_group: str,
        version: int = 1,
    ) -> bool:
        """Check if features exist for an entity.

        Args:
            entity_type: Type of entity.
            entity_id: Unique entity identifier.
            feature_group: Name of the feature group.
            version: Feature group version.

        Returns:
            True if features exist.
        """
        if not self._client:
            raise FeatureStoreConnectionError("Not connected. Call connect() first.")

        key = self._get_key(entity_type, entity_id, feature_group, version)
        return await self._client.exists(key) > 0

    async def get_ttl(
        self,
        entity_type: str,
        entity_id: str,
        feature_group: str,
        version: int = 1,
    ) -> int:
        """Get remaining TTL for features.

        Args:
            entity_type: Type of entity.
            entity_id: Unique entity identifier.
            feature_group: Name of the feature group.
            version: Feature group version.

        Returns:
            Remaining TTL in seconds, -1 if no TTL, -2 if key doesn't exist.
        """
        if not self._client:
            raise FeatureStoreConnectionError("Not connected. Call connect() first.")

        key = self._get_key(entity_type, entity_id, feature_group, version)
        return await self._client.ttl(key)

    async def refresh_ttl(
        self,
        entity_type: str,
        entity_id: str,
        feature_group: str,
        version: int = 1,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Refresh TTL for features.

        Args:
            entity_type: Type of entity.
            entity_id: Unique entity identifier.
            feature_group: Name of the feature group.
            version: Feature group version.
            ttl_seconds: Optional custom TTL.

        Returns:
            True if TTL was set, False if key doesn't exist.
        """
        if not self._client:
            raise FeatureStoreConnectionError("Not connected. Call connect() first.")

        key = self._get_key(entity_type, entity_id, feature_group, version)
        ttl = ttl_seconds or self._get_ttl(feature_group)
        return await self._client.expire(key, ttl)

    async def set_features_batch(
        self,
        entity_type: str,
        features_batch: list[tuple[str, dict[str, Any]]],
        feature_group: str,
        version: int = 1,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store features for multiple entities.

        Args:
            entity_type: Type of entity.
            features_batch: List of (entity_id, features) tuples.
            feature_group: Name of the feature group.
            version: Feature group version.
            ttl_seconds: Optional custom TTL.
        """
        if not self._client:
            raise FeatureStoreConnectionError("Not connected. Call connect() first.")

        ttl = ttl_seconds or self._get_ttl(feature_group)
        updated_at = datetime.now(UTC).isoformat()

        async with self._client.pipeline() as pipe:
            for entity_id, features in features_batch:
                key = self._get_key(entity_type, entity_id, feature_group, version)
                string_features = {k: str(v) for k, v in features.items()}
                string_features["_updated_at"] = updated_at

                await pipe.hset(key, mapping=string_features)
                await pipe.expire(key, ttl)

            await pipe.execute()

    def _deserialize_features(self, data: dict[str, str]) -> dict[str, Any]:
        """Deserialize Redis hash values to Python types.

        Args:
            data: Raw Redis hash data (all strings).

        Returns:
            Dictionary with properly typed values.
        """
        result: dict[str, Any] = {}
        for key, value in data.items():
            if key == "_updated_at":
                result[key] = value
                continue

            # Try to convert to appropriate type
            result[key] = self._parse_value(value)

        return result

    def _parse_value(self, value: str) -> Any:
        """Parse a string value to its appropriate type.

        Args:
            value: String value from Redis.

        Returns:
            Parsed value (int, float, bool, or string).
        """
        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try bool
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Return as string
        return value

    async def __aenter__(self) -> "OnlineStore":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
