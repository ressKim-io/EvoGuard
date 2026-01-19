"""Repository for Feature Registry CRUD operations."""

import hashlib
import json
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ml_service.feature_store.registry.models import (
    FeatureDefinition,
    FeatureGroup,
    FeatureGroupFeature,
    ModelFeatureLineage,
)
from ml_service.feature_store.registry.schemas import (
    FeatureDefinitionCreate,
    FeatureDefinitionUpdate,
    FeatureGroupCreate,
    FeatureGroupUpdate,
    ModelLineageCreate,
)


class FeatureDefinitionRepository:
    """Repository for FeatureDefinition CRUD operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, data: FeatureDefinitionCreate) -> FeatureDefinition:
        """Create a new feature definition."""
        feature = FeatureDefinition(
            name=data.name,
            data_type=data.data_type,
            entity_type=data.entity_type,
            source_type=data.source_type,
            computation_config=data.computation_config,
            description=data.description,
        )
        self.session.add(feature)
        await self.session.commit()
        await self.session.refresh(feature)
        return feature

    async def get_by_id(self, feature_id: UUID) -> FeatureDefinition | None:
        """Get feature definition by ID."""
        result = await self.session.execute(
            select(FeatureDefinition).where(FeatureDefinition.id == feature_id)
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> FeatureDefinition | None:
        """Get feature definition by name."""
        result = await self.session.execute(
            select(FeatureDefinition).where(FeatureDefinition.name == name)
        )
        return result.scalar_one_or_none()

    async def get_by_names(self, names: list[str]) -> list[FeatureDefinition]:
        """Get multiple feature definitions by names."""
        result = await self.session.execute(
            select(FeatureDefinition).where(FeatureDefinition.name.in_(names))
        )
        return list(result.scalars().all())

    async def list_all(
        self,
        entity_type: str | None = None,
        is_active: bool | None = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FeatureDefinition]:
        """List feature definitions with optional filtering."""
        query = select(FeatureDefinition)

        if entity_type:
            query = query.where(FeatureDefinition.entity_type == entity_type)
        if is_active is not None:
            query = query.where(FeatureDefinition.is_active == is_active)

        query = query.order_by(FeatureDefinition.name).limit(limit).offset(offset)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update(
        self, feature_id: UUID, data: FeatureDefinitionUpdate
    ) -> FeatureDefinition | None:
        """Update a feature definition."""
        feature = await self.get_by_id(feature_id)
        if not feature:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(feature, key, value)

        # Increment version on update
        feature.version += 1

        await self.session.commit()
        await self.session.refresh(feature)
        return feature

    async def delete(self, feature_id: UUID) -> bool:
        """Soft delete a feature definition (set is_active=False)."""
        feature = await self.get_by_id(feature_id)
        if not feature:
            return False

        feature.is_active = False
        await self.session.commit()
        return True


class FeatureGroupRepository:
    """Repository for FeatureGroup CRUD operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    def _compute_schema_hash(self, feature_names: list[str]) -> str:
        """Compute a hash of the feature schema for versioning."""
        schema_str = json.dumps(sorted(feature_names), sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    async def create(self, data: FeatureGroupCreate) -> FeatureGroup:
        """Create a new feature group with associated features."""
        # Get feature definitions by name
        feature_repo = FeatureDefinitionRepository(self.session)
        features = await feature_repo.get_by_names(data.feature_names)

        if len(features) != len(data.feature_names):
            found_names = {f.name for f in features}
            missing = set(data.feature_names) - found_names
            raise ValueError(f"Feature definitions not found: {missing}")

        # Create feature group
        schema_hash = self._compute_schema_hash(data.feature_names)
        group = FeatureGroup(
            name=data.name,
            description=data.description,
            entity_type=data.entity_type,
            schema_hash=schema_hash,
        )
        self.session.add(group)
        await self.session.flush()  # Get the group ID

        # Create associations
        for feature in features:
            association = FeatureGroupFeature(
                feature_group_id=group.id,
                feature_definition_id=feature.id,
            )
            self.session.add(association)

        await self.session.commit()
        await self.session.refresh(group)

        # Load features relationship
        result = await self.session.execute(
            select(FeatureGroup)
            .where(FeatureGroup.id == group.id)
            .options(selectinload(FeatureGroup.features).selectinload(FeatureGroupFeature.feature_definition))
        )
        return result.scalar_one()

    async def get_by_id(self, group_id: UUID) -> FeatureGroup | None:
        """Get feature group by ID with features loaded."""
        result = await self.session.execute(
            select(FeatureGroup)
            .where(FeatureGroup.id == group_id)
            .options(selectinload(FeatureGroup.features).selectinload(FeatureGroupFeature.feature_definition))
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> FeatureGroup | None:
        """Get feature group by name with features loaded."""
        result = await self.session.execute(
            select(FeatureGroup)
            .where(FeatureGroup.name == name)
            .options(selectinload(FeatureGroup.features).selectinload(FeatureGroupFeature.feature_definition))
        )
        return result.scalar_one_or_none()

    async def list_all(
        self,
        entity_type: str | None = None,
        is_active: bool | None = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FeatureGroup]:
        """List feature groups with optional filtering."""
        query = select(FeatureGroup)

        if entity_type:
            query = query.where(FeatureGroup.entity_type == entity_type)
        if is_active is not None:
            query = query.where(FeatureGroup.is_active == is_active)

        query = (
            query.options(selectinload(FeatureGroup.features).selectinload(FeatureGroupFeature.feature_definition))
            .order_by(FeatureGroup.name)
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update(self, group_id: UUID, data: FeatureGroupUpdate) -> FeatureGroup | None:
        """Update a feature group."""
        group = await self.get_by_id(group_id)
        if not group:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(group, key, value)

        await self.session.commit()
        await self.session.refresh(group)
        return group

    async def delete(self, group_id: UUID) -> bool:
        """Soft delete a feature group (set is_active=False)."""
        group = await self.get_by_id(group_id)
        if not group:
            return False

        group.is_active = False
        await self.session.commit()
        return True

    def get_feature_definitions(self, group: FeatureGroup) -> list[FeatureDefinition]:
        """Extract feature definitions from a loaded group."""
        return [assoc.feature_definition for assoc in group.features]


class ModelLineageRepository:
    """Repository for ModelFeatureLineage CRUD operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, data: ModelLineageCreate) -> ModelFeatureLineage:
        """Create a new model-feature lineage record."""
        # Get feature group
        group_repo = FeatureGroupRepository(self.session)
        group = await group_repo.get_by_name(data.feature_group_name)
        if not group:
            raise ValueError(f"Feature group not found: {data.feature_group_name}")

        # Create schema snapshot
        features = group_repo.get_feature_definitions(group)
        schema_snapshot = {
            "features": [
                {
                    "name": f.name,
                    "data_type": f.data_type,
                    "entity_type": f.entity_type,
                    "source_type": f.source_type,
                }
                for f in features
            ],
            "version": group.version,
            "schema_hash": group.schema_hash,
        }

        lineage = ModelFeatureLineage(
            mlflow_run_id=data.mlflow_run_id,
            mlflow_model_name=data.mlflow_model_name,
            mlflow_model_version=data.mlflow_model_version,
            feature_group_id=group.id,
            feature_group_version=group.version,
            feature_schema_snapshot=schema_snapshot,
        )
        self.session.add(lineage)
        await self.session.commit()
        await self.session.refresh(lineage)
        return lineage

    async def get_by_run_id(self, run_id: str) -> list[ModelFeatureLineage]:
        """Get lineage records by MLflow run ID."""
        result = await self.session.execute(
            select(ModelFeatureLineage)
            .where(ModelFeatureLineage.mlflow_run_id == run_id)
            .options(selectinload(ModelFeatureLineage.feature_group))
        )
        return list(result.scalars().all())

    async def get_by_model(
        self, model_name: str, model_version: int | None = None
    ) -> list[ModelFeatureLineage]:
        """Get lineage records by model name and optional version."""
        query = select(ModelFeatureLineage).where(
            ModelFeatureLineage.mlflow_model_name == model_name
        )
        if model_version is not None:
            query = query.where(ModelFeatureLineage.mlflow_model_version == model_version)

        query = query.options(selectinload(ModelFeatureLineage.feature_group))
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_schema_for_model(
        self, model_name: str, model_version: int
    ) -> dict[str, Any] | None:
        """Get the feature schema snapshot for a specific model version."""
        result = await self.session.execute(
            select(ModelFeatureLineage)
            .where(
                ModelFeatureLineage.mlflow_model_name == model_name,
                ModelFeatureLineage.mlflow_model_version == model_version,
            )
            .order_by(ModelFeatureLineage.created_at.desc())
            .limit(1)
        )
        lineage = result.scalar_one_or_none()
        return lineage.feature_schema_snapshot if lineage else None
