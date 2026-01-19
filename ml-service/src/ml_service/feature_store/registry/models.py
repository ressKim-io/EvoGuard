"""SQLAlchemy ORM models for Feature Registry."""

import uuid
from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    type_annotation_map: ClassVar[dict[type, type]] = {
        dict[str, Any]: JSONB,
    }


class FeatureDefinition(Base):
    """Feature definition metadata."""

    __tablename__ = "feature_definitions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    data_type: Mapped[str] = mapped_column(String(20), nullable=False)  # int, float, string, embedding
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)  # text, battle, user
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)  # computed, raw, aggregated
    computation_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    feature_groups: Mapped[list["FeatureGroupFeature"]] = relationship(
        "FeatureGroupFeature",
        back_populates="feature_definition",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_feature_definitions_entity", "entity_type"),
        Index("idx_feature_definitions_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<FeatureDefinition(name={self.name}, type={self.data_type})>"


class FeatureGroup(Base):
    """Feature group - collection of related features."""

    __tablename__ = "feature_groups"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1)
    schema_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    features: Mapped[list["FeatureGroupFeature"]] = relationship(
        "FeatureGroupFeature",
        back_populates="feature_group",
        cascade="all, delete-orphan",
    )
    lineages: Mapped[list["ModelFeatureLineage"]] = relationship(
        "ModelFeatureLineage",
        back_populates="feature_group",
    )

    __table_args__ = (
        Index("idx_feature_groups_entity", "entity_type"),
        Index("idx_feature_groups_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<FeatureGroup(name={self.name}, entity_type={self.entity_type})>"


class FeatureGroupFeature(Base):
    """Many-to-many relationship between FeatureGroup and FeatureDefinition."""

    __tablename__ = "feature_group_features"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    feature_group_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("feature_groups.id", ondelete="CASCADE"),
        nullable=False,
    )
    feature_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("feature_definitions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Relationships
    feature_group: Mapped["FeatureGroup"] = relationship(
        "FeatureGroup",
        back_populates="features",
    )
    feature_definition: Mapped["FeatureDefinition"] = relationship(
        "FeatureDefinition",
        back_populates="feature_groups",
    )

    __table_args__ = (
        Index(
            "idx_feature_group_features_unique",
            "feature_group_id",
            "feature_definition_id",
            unique=True,
        ),
    )


class ModelFeatureLineage(Base):
    """Model-Feature lineage tracking."""

    __tablename__ = "model_feature_lineage"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    mlflow_run_id: Mapped[str] = mapped_column(String(100), nullable=False)
    mlflow_model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    mlflow_model_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feature_group_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("feature_groups.id"),
        nullable=False,
    )
    feature_group_version: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_schema_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationships
    feature_group: Mapped["FeatureGroup"] = relationship(
        "FeatureGroup",
        back_populates="lineages",
    )

    __table_args__ = (
        Index("idx_lineage_run", "mlflow_run_id"),
        Index("idx_lineage_model", "mlflow_model_name", "mlflow_model_version"),
    )

    def __repr__(self) -> str:
        return f"<ModelFeatureLineage(run_id={self.mlflow_run_id})>"
