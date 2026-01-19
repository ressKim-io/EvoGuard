"""Database connection and session management for Feature Registry."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ml_service.core.exceptions import DatabaseNotInitializedError
from ml_service.feature_store.registry.models import Base

# Global engine and session factory
_engine = None
_session_factory = None


def init_database(database_url: str) -> None:
    """Initialize the database engine and session factory.

    Args:
        database_url: PostgreSQL connection URL (async format).
            Example: postgresql+asyncpg://user:pass@host:5432/dbname
    """
    global _engine, _session_factory

    _engine = create_async_engine(
        database_url,
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def create_tables() -> None:
    """Create all tables in the database."""
    if _engine is None:
        raise DatabaseNotInitializedError()

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables() -> None:
    """Drop all tables in the database."""
    if _engine is None:
        raise DatabaseNotInitializedError()

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session as an async context manager.

    Yields:
        AsyncSession instance.

    Example:
        async with get_session() as session:
            repo = FeatureDefinitionRepository(session)
            feature = await repo.get_by_name("my_feature")
    """
    if _session_factory is None:
        raise DatabaseNotInitializedError()

    session = _session_factory()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions.

    Yields:
        AsyncSession instance.

    Example:
        @router.get("/features")
        async def list_features(session: AsyncSession = Depends(get_session_dependency)):
            ...
    """
    if _session_factory is None:
        raise DatabaseNotInitializedError()

    session = _session_factory()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def close_database() -> None:
    """Close the database connection pool."""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
