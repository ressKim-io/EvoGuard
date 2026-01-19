"""Feature Registry - PostgreSQL metadata storage."""

from ml_service.feature_store.registry.database import (
    close_database,
    create_tables,
    get_session,
    get_session_dependency,
    init_database,
)
from ml_service.feature_store.registry.models import (
    Base,
    FeatureDefinition,
    FeatureGroup,
    FeatureGroupFeature,
    ModelFeatureLineage,
)
from ml_service.feature_store.registry.repository import (
    FeatureDefinitionRepository,
    FeatureGroupRepository,
    ModelLineageRepository,
)
from ml_service.feature_store.registry.schemas import (
    FeatureDefinitionCreate,
    FeatureDefinitionResponse,
    FeatureDefinitionUpdate,
    FeatureGroupCreate,
    FeatureGroupListResponse,
    FeatureGroupResponse,
    FeatureGroupUpdate,
    ModelLineageCreate,
    ModelLineageResponse,
)

__all__ = [
    "Base",
    "FeatureDefinition",
    "FeatureDefinitionCreate",
    "FeatureDefinitionRepository",
    "FeatureDefinitionResponse",
    "FeatureDefinitionUpdate",
    "FeatureGroup",
    "FeatureGroupCreate",
    "FeatureGroupFeature",
    "FeatureGroupListResponse",
    "FeatureGroupRepository",
    "FeatureGroupResponse",
    "FeatureGroupUpdate",
    "ModelFeatureLineage",
    "ModelLineageCreate",
    "ModelLineageRepository",
    "ModelLineageResponse",
    "close_database",
    "create_tables",
    "get_session",
    "get_session_dependency",
    "init_database",
]
