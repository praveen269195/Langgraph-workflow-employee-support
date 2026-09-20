"""
Migration — creates all application tables on startup.

document_chunks is managed separately via VectorRepository (pgvector ORM)
because it lives on a different declarative Base.  The migration drops and
recreates document_chunks every time the app starts so that schema changes
(e.g. adding page / chunk_index columns) are always applied without a
manual ALTER TABLE.

All other tables (logger, employees, sessions, escalations) use CREATE IF
NOT EXISTS — they are never dropped automatically.
"""

from sqlalchemy import inspect, text

from repositories.Database import Database
from repositories.schemas.schema import (
    Employee,
    ErrorLogger,
    Escalation,
    Session,
)
from repositories.vector_repository import VectorRepository
from utils.logger import Logger

# Ordered by FK dependency — parent tables before child tables
TABLE_ORDER_CREATION: list[str] = [
    ErrorLogger.__tablename__,
    Employee.__tablename__,
    Session.__tablename__,
    Escalation.__tablename__,
]

MODEL_CLASSES: dict = {
    ErrorLogger.__tablename__: ErrorLogger,
    Employee.__tablename__:    Employee,
    Session.__tablename__:     Session,
    Escalation.__tablename__:  Escalation,
}


class Migration:
    def __init__(self) -> None:
        self.db           = Database()
        self.engine       = self.db.engine
        self.logger       = Logger("migration")
        self.vector_repo  = VectorRepository()

    async def create_tables(self) -> None:
        """
        Full startup migration sequence:
          1. Ensure pgvector + pgcrypto extensions exist.
          2. DROP document_chunks (recreate on every startup to pick up schema changes).
          3. CREATE document_chunks via VectorRepository.create_table().
          4. CREATE IF NOT EXISTS for all other application tables.
        """
        self.logger.info("create_tables started")

        # ── 1. Extensions ──────────────────────────────────────────────────────
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        self.logger.info("create_tables: extensions ensured")

        # ── 2 + 3. document_chunks: safe schema migration ──────────────────────
        # Use CREATE IF NOT EXISTS — never drop data on restart.
        # If the table already exists but is missing the new columns (page,
        # chunk_index), add them with ALTER TABLE ... ADD COLUMN IF NOT EXISTS.
        # This is a one-time operation; once the columns exist it is a no-op.
        self.logger.info("create_tables: ensuring document_chunks schema")
        await self.vector_repo.create_table()

        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    "ALTER TABLE document_chunks "
                    "ADD COLUMN IF NOT EXISTS page INTEGER"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE document_chunks "
                    "ADD COLUMN IF NOT EXISTS chunk_index INTEGER NOT NULL DEFAULT 0"
                )
            )
        self.logger.info("create_tables: document_chunks schema up to date")

        # ── 4. Application tables (CREATE IF NOT EXISTS) ───────────────────────
        async with self.engine.begin() as conn:
            for table_name in TABLE_ORDER_CREATION:
                table_exists: bool = await conn.run_sync(
                    lambda sync_conn, tn=table_name: inspect(sync_conn).has_table(tn)
                )
                if not table_exists:
                    self.logger.info(f"create_tables: creating table '{table_name}'")
                    await conn.run_sync(MODEL_CLASSES[table_name].__table__.create)
                else:
                    self.logger.info(
                        f"create_tables: table '{table_name}' already exists — skipping"
                    )

        self.logger.info("create_tables ended")
