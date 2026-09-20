from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from settings import config


class Database:
    """Singleton async database engine and session factory.

    Instantiate once at startup (via lifespan) and share the single instance
    across the application.  Never open a new engine per request.
    """

    _instance: "Database | None" = None

    def __new__(cls) -> "Database":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.engine = create_async_engine(
            config.async_db_uri(),
            pool_pre_ping=True,   # detect stale connections
            pool_size=10,
            max_overflow=20,
        )
        self.SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        self._initialized = True

    # ── Session helper ─────────────────────────────────────────────────────────
    @asynccontextmanager
    async def get_session(self):
        """Async context manager for a database session.

        Usage::

            async with self.db.get_session() as session:
                session.add(obj)
                # commit is handled automatically on clean exit
        """
        async with self.SessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # ── Health check ───────────────────────────────────────────────────────────
    async def test_connection(self) -> bool:
        """Return True when a basic SELECT 1 succeeds."""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
