"""Tests for Online Store (Redis)."""

import pytest
from fakeredis import aioredis as fakeredis

from ml_service.core.exceptions import FeatureStoreConnectionError
from ml_service.feature_store.online import OnlineStore, OnlineStoreConfig


class TestOnlineStore:
    """Tests for OnlineStore."""

    @pytest.fixture
    async def store(self) -> OnlineStore:
        """Create a store with fake Redis."""
        store = OnlineStore(OnlineStoreConfig())
        # Replace with fake redis
        store._client = fakeredis.FakeRedis(decode_responses=True)
        return store

    async def test_set_and_get_features(self, store: OnlineStore) -> None:
        """Test setting and getting features."""
        features = {
            "text_length": 100,
            "word_count": 20,
            "unicode_ratio": 0.15,
        }

        await store.set_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
            features=features,
        )

        result = await store.get_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
        )

        assert result is not None
        assert result["text_length"] == 100
        assert result["word_count"] == 20
        assert result["unicode_ratio"] == 0.15
        assert "_updated_at" in result

    async def test_get_features_not_found(self, store: OnlineStore) -> None:
        """Test getting non-existent features."""
        result = await store.get_features(
            entity_type="text",
            entity_id="nonexistent",
            feature_group="text_features",
        )

        assert result is None

    async def test_get_features_batch(self, store: OnlineStore) -> None:
        """Test batch feature retrieval."""
        # Set features for multiple entities
        for i in range(1, 4):
            await store.set_features(
                entity_type="text",
                entity_id=f"entity-{i}",
                feature_group="text_features",
                features={"text_length": 100 * i},
            )

        results = await store.get_features_batch(
            entity_type="text",
            entity_ids=["entity-1", "entity-2", "entity-3", "entity-4"],
            feature_group="text_features",
        )

        assert len(results) == 4
        assert results["entity-1"]["text_length"] == 100
        assert results["entity-2"]["text_length"] == 200
        assert results["entity-3"]["text_length"] == 300
        assert results["entity-4"] is None  # Doesn't exist

    async def test_set_features_batch(self, store: OnlineStore) -> None:
        """Test batch feature storage."""
        batch = [
            ("entity-1", {"text_length": 100, "word_count": 20}),
            ("entity-2", {"text_length": 200, "word_count": 40}),
            ("entity-3", {"text_length": 300, "word_count": 60}),
        ]

        await store.set_features_batch(
            entity_type="text",
            features_batch=batch,
            feature_group="text_features",
        )

        # Verify all were stored
        for entity_id, expected_features in batch:
            result = await store.get_features(
                entity_type="text",
                entity_id=entity_id,
                feature_group="text_features",
            )
            assert result is not None
            assert result["text_length"] == expected_features["text_length"]

    async def test_delete_features(self, store: OnlineStore) -> None:
        """Test deleting features."""
        await store.set_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
            features={"text_length": 100},
        )

        # Verify exists
        assert await store.exists(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
        )

        # Delete
        deleted = await store.delete_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
        )
        assert deleted is True

        # Verify deleted
        assert not await store.exists(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
        )

    async def test_delete_features_not_found(self, store: OnlineStore) -> None:
        """Test deleting non-existent features."""
        deleted = await store.delete_features(
            entity_type="text",
            entity_id="nonexistent",
            feature_group="text_features",
        )
        assert deleted is False

    async def test_exists(self, store: OnlineStore) -> None:
        """Test checking feature existence."""
        assert not await store.exists(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
        )

        await store.set_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
            features={"text_length": 100},
        )

        assert await store.exists(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
        )

    async def test_version_isolation(self, store: OnlineStore) -> None:
        """Test that different versions are isolated."""
        await store.set_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
            features={"text_length": 100},
            version=1,
        )

        await store.set_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
            features={"text_length": 200},
            version=2,
        )

        v1 = await store.get_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
            version=1,
        )

        v2 = await store.get_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
            version=2,
        )

        assert v1["text_length"] == 100
        assert v2["text_length"] == 200

    async def test_feature_group_isolation(self, store: OnlineStore) -> None:
        """Test that different feature groups are isolated."""
        await store.set_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
            features={"text_length": 100},
        )

        await store.set_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="battle_features",
            features={"detection_rate": 0.95},
        )

        text_features = await store.get_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
        )

        battle_features = await store.get_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="battle_features",
        )

        assert "text_length" in text_features
        assert "detection_rate" in battle_features

    async def test_key_structure(self, store: OnlineStore) -> None:
        """Test key generation."""
        key = store._get_key(
            entity_type="text",
            entity_id="123",
            feature_group="text_features",
            version=1,
        )
        assert key == "feature:text:123:text_features:v1"

    async def test_type_parsing(self, store: OnlineStore) -> None:
        """Test that values are parsed to correct types."""
        await store.set_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
            features={
                "int_val": 100,
                "float_val": 0.15,
                "bool_val": True,
                "str_val": "hello",
            },
        )

        result = await store.get_features(
            entity_type="text",
            entity_id="entity-1",
            feature_group="text_features",
        )

        assert isinstance(result["int_val"], int)
        assert isinstance(result["float_val"], float)
        assert isinstance(result["bool_val"], bool)
        assert isinstance(result["str_val"], str)

    async def test_not_connected_raises(self) -> None:
        """Test that operations fail when not connected."""
        store = OnlineStore(OnlineStoreConfig())
        # Don't connect

        with pytest.raises(FeatureStoreConnectionError, match="Not connected"):
            await store.get_features(
                entity_type="text",
                entity_id="entity-1",
                feature_group="text_features",
            )


class TestOnlineStoreConfig:
    """Tests for OnlineStoreConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = OnlineStoreConfig()

        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.ssl is False
        assert config.key_prefix == "feature"
        assert config.default_ttl_seconds == 86400

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = OnlineStoreConfig(
            host="redis.example.com",
            port=6380,
            password="secret",
            ssl=True,
        )

        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.password == "secret"
        assert config.ssl is True
