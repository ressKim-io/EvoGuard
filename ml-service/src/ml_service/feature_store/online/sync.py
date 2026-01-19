"""Offline to Online Store synchronization."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ml_service.core.exceptions import FeatureStoreConnectionError
from ml_service.feature_store.offline import OfflineStoreReader
from ml_service.feature_store.online import OnlineStore, OnlineStoreConfig

logger = logging.getLogger(__name__)


class FeatureSync:
    """Synchronize features from Offline Store (Parquet) to Online Store (Redis).

    This class handles the materialization of features from the offline store
    to the online store for real-time serving.

    Example:
        >>> sync = FeatureSync(
        ...     offline_path="/data/features",
        ...     online_config=OnlineStoreConfig(),
        ... )
        >>> await sync.connect()
        >>> stats = await sync.sync_latest_features(
        ...     feature_group="text_features",
        ...     entity_type="text",
        ... )
        >>> print(stats)
        {'synced': 100, 'skipped': 5, 'errors': 0}
    """

    def __init__(
        self,
        offline_path: str | Path,
        online_config: OnlineStoreConfig | None = None,
    ) -> None:
        """Initialize the feature sync.

        Args:
            offline_path: Path to offline feature store.
            online_config: Configuration for online store connection.
        """
        self.offline_reader = OfflineStoreReader(offline_path)
        self.online_store = OnlineStore(online_config)
        self._connected = False

    async def connect(self) -> None:
        """Connect to the online store."""
        await self.online_store.connect()
        self._connected = True

    async def close(self) -> None:
        """Close connections."""
        await self.online_store.close()
        self.offline_reader.close()
        self._connected = False

    async def sync_latest_features(
        self,
        feature_group: str,
        entity_type: str,
        version: int = 1,
        batch_size: int = 100,
        ttl_seconds: int | None = None,
    ) -> dict[str, int]:
        """Sync the latest features for all entities to the online store.

        Reads the most recent features for each entity from the offline store
        and writes them to the online store.

        Args:
            feature_group: Name of the feature group.
            entity_type: Entity type for online store keys.
            version: Feature group version.
            batch_size: Number of features to sync per batch.
            ttl_seconds: Optional custom TTL for online store.

        Returns:
            Dictionary with sync statistics (synced, skipped, errors).
        """
        if not self._connected:
            raise FeatureStoreConnectionError("Not connected. Call connect() first.")

        stats = {"synced": 0, "skipped": 0, "errors": 0}

        try:
            # Get all features from offline store
            table = self.offline_reader.get_features(
                feature_group=feature_group,
                version=version,
            )

            # Convert to list of dicts
            records = table.to_pylist()

            if not records:
                logger.info(f"No features to sync for {feature_group} v{version}")
                return stats

            # Group by entity_id and get latest for each
            entity_features: dict[str, dict[str, Any]] = {}
            for record in records:
                entity_id = record.get("entity_id")
                if entity_id:
                    # Keep the latest (last occurrence in sorted data)
                    entity_features[entity_id] = record

            logger.info(
                f"Syncing {len(entity_features)} entities for {feature_group} v{version}"
            )

            # Sync in batches
            entity_ids = list(entity_features.keys())
            for i in range(0, len(entity_ids), batch_size):
                batch_ids = entity_ids[i : i + batch_size]
                batch_data = [
                    (eid, self._prepare_features(entity_features[eid]))
                    for eid in batch_ids
                ]

                try:
                    await self.online_store.set_features_batch(
                        entity_type=entity_type,
                        features_batch=batch_data,
                        feature_group=feature_group,
                        version=version,
                        ttl_seconds=ttl_seconds,
                    )
                    stats["synced"] += len(batch_data)
                except Exception as e:
                    logger.error(f"Error syncing batch: {e}")
                    stats["errors"] += len(batch_data)

            logger.info(f"Sync completed: {stats}")

        except FileNotFoundError:
            logger.warning(f"No offline data found for {feature_group} v{version}")
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            raise

        return stats

    async def sync_entity_features(
        self,
        feature_group: str,
        entity_type: str,
        entity_ids: list[str],
        version: int = 1,
        ttl_seconds: int | None = None,
    ) -> dict[str, int]:
        """Sync features for specific entities.

        Args:
            feature_group: Name of the feature group.
            entity_type: Entity type for online store keys.
            entity_ids: List of entity IDs to sync.
            version: Feature group version.
            ttl_seconds: Optional custom TTL.

        Returns:
            Dictionary with sync statistics.
        """
        if not self._connected:
            raise FeatureStoreConnectionError("Not connected. Call connect() first.")

        stats = {"synced": 0, "skipped": 0, "errors": 0}

        try:
            # Get latest features for specific entities
            table = self.offline_reader.get_latest_features(
                feature_group=feature_group,
                entity_ids=entity_ids,
                version=version,
            )

            records = table.to_pylist()

            if not records:
                logger.info("No features found for specified entities")
                stats["skipped"] = len(entity_ids)
                return stats

            # Prepare batch data
            found_ids = set()
            batch_data = []
            for record in records:
                entity_id = record.get("entity_id")
                if entity_id:
                    found_ids.add(entity_id)
                    batch_data.append((entity_id, self._prepare_features(record)))

            # Sync to online store
            if batch_data:
                await self.online_store.set_features_batch(
                    entity_type=entity_type,
                    features_batch=batch_data,
                    feature_group=feature_group,
                    version=version,
                    ttl_seconds=ttl_seconds,
                )
                stats["synced"] = len(batch_data)

            # Count skipped (not found in offline store)
            stats["skipped"] = len(set(entity_ids) - found_ids)

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            stats["errors"] = len(entity_ids)

        return stats

    async def refresh_online_cache(
        self,
        feature_group: str,
        entity_type: str,
        entity_ids: list[str],
        version: int = 1,
        ttl_seconds: int | None = None,
    ) -> int:
        """Refresh TTL for existing online features or sync if missing.

        Args:
            feature_group: Name of the feature group.
            entity_type: Entity type.
            entity_ids: List of entity IDs.
            version: Feature group version.
            ttl_seconds: Optional custom TTL.

        Returns:
            Number of entities refreshed or synced.
        """
        if not self._connected:
            raise FeatureStoreConnectionError("Not connected. Call connect() first.")

        refreshed = 0
        missing_ids = []

        # Check which entities exist in online store
        for entity_id in entity_ids:
            exists = await self.online_store.exists(
                entity_type=entity_type,
                entity_id=entity_id,
                feature_group=feature_group,
                version=version,
            )

            if exists:
                # Refresh TTL
                await self.online_store.refresh_ttl(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    feature_group=feature_group,
                    version=version,
                    ttl_seconds=ttl_seconds,
                )
                refreshed += 1
            else:
                missing_ids.append(entity_id)

        # Sync missing entities from offline store
        if missing_ids:
            stats = await self.sync_entity_features(
                feature_group=feature_group,
                entity_type=entity_type,
                entity_ids=missing_ids,
                version=version,
                ttl_seconds=ttl_seconds,
            )
            refreshed += stats["synced"]

        return refreshed

    def _prepare_features(self, record: dict[str, Any]) -> dict[str, Any]:
        """Prepare features for online store by removing metadata fields.

        Args:
            record: Raw feature record from offline store.

        Returns:
            Cleaned feature dictionary.
        """
        # Fields to exclude from online store
        exclude_fields = {"entity_id", "event_timestamp", "created_at", "rn"}

        return {
            k: self._serialize_value(v)
            for k, v in record.items()
            if k not in exclude_fields and v is not None
        }

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for Redis storage.

        Args:
            value: Value to serialize.

        Returns:
            Serialized value (converted datetime to ISO string).
        """
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    async def __aenter__(self) -> "FeatureSync":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
