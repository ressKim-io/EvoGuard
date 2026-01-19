"""Pydantic schemas for Feature Registry API."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Feature Definition Schemas
# =============================================================================


class FeatureDefinitionCreate(BaseModel):
    """Schema for creating a feature definition."""

    name: str = Field(..., min_length=1, max_length=100, description="Unique feature name")
    data_type: str = Field(
        ...,
        pattern="^(int|float|string|bool|embedding)$",
        description="Data type: int, float, string, bool, embedding",
    )
    entity_type: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Entity type: text, battle, user",
    )
    source_type: str = Field(
        ...,
        pattern="^(computed|raw|aggregated)$",
        description="Source type: computed, raw, aggregated",
    )
    computation_config: dict[str, Any] | None = Field(
        default=None,
        description="Configuration for feature computation",
    )
    description: str | None = Field(default=None, description="Feature description")


class FeatureDefinitionUpdate(BaseModel):
    """Schema for updating a feature definition."""

    description: str | None = None
    computation_config: dict[str, Any] | None = None
    is_active: bool | None = None


class FeatureDefinitionResponse(BaseModel):
    """Schema for feature definition response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    data_type: str
    entity_type: str
    source_type: str
    computation_config: dict[str, Any] | None
    description: str | None
    version: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Feature Group Schemas
# =============================================================================


class FeatureGroupCreate(BaseModel):
    """Schema for creating a feature group."""

    name: str = Field(..., min_length=1, max_length=100, description="Unique group name")
    description: str | None = Field(default=None, description="Group description")
    entity_type: str = Field(..., min_length=1, max_length=50, description="Entity type")
    feature_names: list[str] = Field(
        ...,
        min_length=1,
        description="List of feature names to include in this group",
    )


class FeatureGroupUpdate(BaseModel):
    """Schema for updating a feature group."""

    description: str | None = None
    is_active: bool | None = None


class FeatureGroupResponse(BaseModel):
    """Schema for feature group response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    entity_type: str
    version: int
    schema_hash: str | None
    is_active: bool
    features: list[FeatureDefinitionResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class FeatureGroupListResponse(BaseModel):
    """Schema for feature group list response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    entity_type: str
    version: int
    is_active: bool
    feature_count: int = 0


# =============================================================================
# Model Lineage Schemas
# =============================================================================


class ModelLineageCreate(BaseModel):
    """Schema for creating model-feature lineage."""

    mlflow_run_id: str = Field(..., min_length=1, max_length=100)
    mlflow_model_name: str | None = Field(default=None, max_length=100)
    mlflow_model_version: int | None = Field(default=None, ge=1)
    feature_group_name: str = Field(..., min_length=1)


class ModelLineageResponse(BaseModel):
    """Schema for model lineage response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    mlflow_run_id: str
    mlflow_model_name: str | None
    mlflow_model_version: int | None
    feature_group_id: UUID
    feature_group_version: int
    feature_schema_snapshot: dict[str, Any]
    created_at: datetime


# =============================================================================
# Feature Computation Schemas
# =============================================================================


class ComputeFeaturesRequest(BaseModel):
    """Schema for computing features."""

    feature_group: str = Field(..., description="Feature group name")
    entity_ids: list[str] = Field(..., min_length=1, description="Entity IDs")
    data: list[dict[str, Any]] = Field(..., min_length=1, description="Raw data for computation")


class ComputedFeature(BaseModel):
    """Schema for a single computed feature result."""

    entity_id: str
    features: dict[str, Any]


class ComputeFeaturesResponse(BaseModel):
    """Schema for computed features response."""

    feature_group: str
    features: list[ComputedFeature]
    computed_at: datetime
