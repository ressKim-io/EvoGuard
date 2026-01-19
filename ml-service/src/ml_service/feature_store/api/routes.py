"""FastAPI routes for Feature Store."""

from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from ml_service.feature_store.compute.text_features import TextFeatureTransformer
from ml_service.feature_store.registry import (
    FeatureDefinitionCreate,
    FeatureDefinitionRepository,
    FeatureDefinitionResponse,
    FeatureDefinitionUpdate,
    FeatureGroupCreate,
    FeatureGroupListResponse,
    FeatureGroupRepository,
    FeatureGroupResponse,
    FeatureGroupUpdate,
    ModelLineageCreate,
    ModelLineageRepository,
    ModelLineageResponse,
    get_session_dependency,
)
from ml_service.feature_store.registry.schemas import (
    ComputedFeature,
    ComputeFeaturesRequest,
    ComputeFeaturesResponse,
)

router = APIRouter(prefix="/features", tags=["features"])

# Type alias for database session dependency
SessionDep = Annotated[AsyncSession, Depends(get_session_dependency)]


# =============================================================================
# Feature Definition Endpoints
# =============================================================================


@router.post(
    "/definitions",
    response_model=FeatureDefinitionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a feature definition",
)
async def create_feature_definition(
    data: FeatureDefinitionCreate,
    session: SessionDep,
) -> FeatureDefinitionResponse:
    """Create a new feature definition."""
    repo = FeatureDefinitionRepository(session)

    # Check if name already exists
    existing = await repo.get_by_name(data.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Feature definition with name '{data.name}' already exists",
        )

    feature = await repo.create(data)
    return FeatureDefinitionResponse.model_validate(feature)


@router.get(
    "/definitions",
    response_model=list[FeatureDefinitionResponse],
    summary="List feature definitions",
)
async def list_feature_definitions(
    session: SessionDep,
    entity_type: str | None = Query(default=None, description="Filter by entity type"),
    is_active: bool | None = Query(default=True, description="Filter by active status"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> list[FeatureDefinitionResponse]:
    """List feature definitions with optional filtering."""
    repo = FeatureDefinitionRepository(session)
    features = await repo.list_all(
        entity_type=entity_type,
        is_active=is_active,
        limit=limit,
        offset=offset,
    )
    return [FeatureDefinitionResponse.model_validate(f) for f in features]


@router.get(
    "/definitions/{feature_id}",
    response_model=FeatureDefinitionResponse,
    summary="Get a feature definition",
)
async def get_feature_definition(
    feature_id: UUID,
    session: SessionDep,
) -> FeatureDefinitionResponse:
    """Get a feature definition by ID."""
    repo = FeatureDefinitionRepository(session)
    feature = await repo.get_by_id(feature_id)
    if not feature:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature definition not found: {feature_id}",
        )
    return FeatureDefinitionResponse.model_validate(feature)


@router.patch(
    "/definitions/{feature_id}",
    response_model=FeatureDefinitionResponse,
    summary="Update a feature definition",
)
async def update_feature_definition(
    feature_id: UUID,
    data: FeatureDefinitionUpdate,
    session: SessionDep,
) -> FeatureDefinitionResponse:
    """Update a feature definition."""
    repo = FeatureDefinitionRepository(session)
    feature = await repo.update(feature_id, data)
    if not feature:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature definition not found: {feature_id}",
        )
    return FeatureDefinitionResponse.model_validate(feature)


@router.delete(
    "/definitions/{feature_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a feature definition",
)
async def delete_feature_definition(
    feature_id: UUID,
    session: SessionDep,
) -> None:
    """Soft delete a feature definition (set is_active=False)."""
    repo = FeatureDefinitionRepository(session)
    deleted = await repo.delete(feature_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature definition not found: {feature_id}",
        )


# =============================================================================
# Feature Group Endpoints
# =============================================================================


@router.post(
    "/groups",
    response_model=FeatureGroupResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a feature group",
)
async def create_feature_group(
    data: FeatureGroupCreate,
    session: SessionDep,
) -> FeatureGroupResponse:
    """Create a new feature group."""
    repo = FeatureGroupRepository(session)

    # Check if name already exists
    existing = await repo.get_by_name(data.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Feature group with name '{data.name}' already exists",
        )

    try:
        group = await repo.create(data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    # Convert to response with features
    features = repo.get_feature_definitions(group)
    return FeatureGroupResponse(
        id=group.id,
        name=group.name,
        description=group.description,
        entity_type=group.entity_type,
        version=group.version,
        schema_hash=group.schema_hash,
        is_active=group.is_active,
        features=[FeatureDefinitionResponse.model_validate(f) for f in features],
        created_at=group.created_at,
        updated_at=group.updated_at,
    )


@router.get(
    "/groups",
    response_model=list[FeatureGroupListResponse],
    summary="List feature groups",
)
async def list_feature_groups(
    session: SessionDep,
    entity_type: str | None = Query(default=None, description="Filter by entity type"),
    is_active: bool | None = Query(default=True, description="Filter by active status"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> list[FeatureGroupListResponse]:
    """List feature groups with optional filtering."""
    repo = FeatureGroupRepository(session)
    groups = await repo.list_all(
        entity_type=entity_type,
        is_active=is_active,
        limit=limit,
        offset=offset,
    )
    return [
        FeatureGroupListResponse(
            id=g.id,
            name=g.name,
            description=g.description,
            entity_type=g.entity_type,
            version=g.version,
            is_active=g.is_active,
            feature_count=len(g.features),
        )
        for g in groups
    ]


@router.get(
    "/groups/{group_name}",
    response_model=FeatureGroupResponse,
    summary="Get a feature group by name",
)
async def get_feature_group(
    group_name: str,
    session: SessionDep,
) -> FeatureGroupResponse:
    """Get a feature group by name with all features."""
    repo = FeatureGroupRepository(session)
    group = await repo.get_by_name(group_name)
    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature group not found: {group_name}",
        )

    features = repo.get_feature_definitions(group)
    return FeatureGroupResponse(
        id=group.id,
        name=group.name,
        description=group.description,
        entity_type=group.entity_type,
        version=group.version,
        schema_hash=group.schema_hash,
        is_active=group.is_active,
        features=[FeatureDefinitionResponse.model_validate(f) for f in features],
        created_at=group.created_at,
        updated_at=group.updated_at,
    )


@router.patch(
    "/groups/{group_name}",
    response_model=FeatureGroupResponse,
    summary="Update a feature group",
)
async def update_feature_group(
    group_name: str,
    data: FeatureGroupUpdate,
    session: SessionDep,
) -> FeatureGroupResponse:
    """Update a feature group."""
    repo = FeatureGroupRepository(session)
    group = await repo.get_by_name(group_name)
    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature group not found: {group_name}",
        )

    group = await repo.update(group.id, data)
    features = repo.get_feature_definitions(group)  # type: ignore
    return FeatureGroupResponse(
        id=group.id,  # type: ignore
        name=group.name,  # type: ignore
        description=group.description,  # type: ignore
        entity_type=group.entity_type,  # type: ignore
        version=group.version,  # type: ignore
        schema_hash=group.schema_hash,  # type: ignore
        is_active=group.is_active,  # type: ignore
        features=[FeatureDefinitionResponse.model_validate(f) for f in features],
        created_at=group.created_at,  # type: ignore
        updated_at=group.updated_at,  # type: ignore
    )


@router.delete(
    "/groups/{group_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a feature group",
)
async def delete_feature_group(
    group_name: str,
    session: SessionDep,
) -> None:
    """Soft delete a feature group (set is_active=False)."""
    repo = FeatureGroupRepository(session)
    group = await repo.get_by_name(group_name)
    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature group not found: {group_name}",
        )

    await repo.delete(group.id)


# =============================================================================
# Feature Computation Endpoints
# =============================================================================


# Registry of available transformers
_TRANSFORMERS: dict[str, Any] = {
    "text_features": TextFeatureTransformer,
}


@router.post(
    "/compute",
    response_model=ComputeFeaturesResponse,
    summary="Compute features for data",
)
async def compute_features(
    request: ComputeFeaturesRequest,
) -> ComputeFeaturesResponse:
    """Compute features for the given data using the specified feature group.

    Note: This endpoint uses in-memory transformers and does not require
    the feature registry database. For production, features should be
    registered in the registry.
    """
    # Get transformer for the feature group
    transformer_class = _TRANSFORMERS.get(request.feature_group)
    if not transformer_class:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown feature group: {request.feature_group}. "
            f"Available: {list(_TRANSFORMERS.keys())}",
        )

    transformer = transformer_class()

    # Validate data length matches entity_ids
    if len(request.entity_ids) != len(request.data):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"entity_ids length ({len(request.entity_ids)}) must match "
            f"data length ({len(request.data)})",
        )

    # Compute features
    computed = []
    for entity_id, data in zip(request.entity_ids, request.data, strict=True):
        # For text features, expect data to have a 'text' field
        if request.feature_group == "text_features":
            text = data.get("text", "")
            features = transformer.transform(text)
        else:
            features = transformer.transform(data)

        computed.append(ComputedFeature(entity_id=entity_id, features=features))

    return ComputeFeaturesResponse(
        feature_group=request.feature_group,
        features=computed,
        computed_at=datetime.now(UTC),
    )


# =============================================================================
# Model Lineage Endpoints
# =============================================================================


@router.post(
    "/lineage",
    response_model=ModelLineageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record model-feature lineage",
)
async def create_model_lineage(
    data: ModelLineageCreate,
    session: SessionDep,
) -> ModelLineageResponse:
    """Record the lineage between a model and its feature group."""
    repo = ModelLineageRepository(session)
    try:
        lineage = await repo.create(data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    return ModelLineageResponse.model_validate(lineage)


@router.get(
    "/lineage/run/{run_id}",
    response_model=list[ModelLineageResponse],
    summary="Get lineage by MLflow run ID",
)
async def get_lineage_by_run(
    run_id: str,
    session: SessionDep,
) -> list[ModelLineageResponse]:
    """Get model-feature lineage records by MLflow run ID."""
    repo = ModelLineageRepository(session)
    lineages = await repo.get_by_run_id(run_id)
    return [ModelLineageResponse.model_validate(lineage) for lineage in lineages]


@router.get(
    "/lineage/model/{model_name}",
    response_model=list[ModelLineageResponse],
    summary="Get lineage by model name",
)
async def get_lineage_by_model(
    model_name: str,
    session: SessionDep,
    model_version: int | None = Query(default=None, description="Filter by model version"),
) -> list[ModelLineageResponse]:
    """Get model-feature lineage records by model name and optional version."""
    repo = ModelLineageRepository(session)
    lineages = await repo.get_by_model(model_name, model_version)
    return [ModelLineageResponse.model_validate(lineage) for lineage in lineages]
