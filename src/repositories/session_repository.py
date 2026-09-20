from datetime import datetime, timezone

from sqlalchemy import select, update

from repositories.Database import Database
from repositories.schemas.schema import Session as SessionModel
from utils.Exceptions.errorcodes import database_error, forbidden_error, not_found_error
from utils.logger import Logger


class SessionRepository:
    """All database operations for the sessions table."""

    def __init__(self) -> None:
        self.db = Database()
        self.logger = Logger("session_repository")

    # ── Create ─────────────────────────────────────────────────────────────────
    async def create_session(self, session_id: str, employee_id: str) -> SessionModel:
        """Insert a new session row and return it."""
        try:
            async with self.db.get_session() as db_session:
                row = SessionModel(
                    session_id=session_id,
                    employee_id=employee_id,
                    last_active_at=datetime.now(timezone.utc),
                    created_by=employee_id,
                    updated_by=employee_id,
                )
                db_session.add(row)
                await db_session.flush()   # populate server defaults before return
                self.logger.info(
                    f"Session created",
                    session_id=session_id,
                    employee_id=employee_id,
                )
                return row
        except Exception as e:
            await self.logger.error("Failed to create session", e)
            raise database_error(str(e))

    # ── Read ───────────────────────────────────────────────────────────────────
    async def get_session_by_id(self, session_id: str) -> SessionModel:
        """Return the session row or raise 404."""
        try:
            async with self.db.get_session() as db_session:
                result = await db_session.execute(
                    select(SessionModel).where(SessionModel.session_id == session_id)
                )
                row = result.scalar_one_or_none()
                if row is None:
                    raise not_found_error(f"session_id '{session_id}'")
                return row
        except Exception as e:
            if hasattr(e, "http_status"):
                raise
            await self.logger.error("Failed to fetch session", e)
            raise database_error(str(e))

    async def validate_owner(self, session_id: str, employee_id: str) -> SessionModel:
        """Return the session if employee_id matches; raise 403 otherwise."""
        row = await self.get_session_by_id(session_id)
        if row.employee_id != employee_id:
            raise forbidden_error()
        return row

    # ── Update last_active_at ──────────────────────────────────────────────────
    async def touch_session(self, session_id: str, employee_id: str) -> None:
        """Bump last_active_at to now."""
        try:
            async with self.db.get_session() as db_session:
                await db_session.execute(
                    update(SessionModel)
                    .where(SessionModel.session_id == session_id)
                    .values(
                        last_active_at=datetime.now(timezone.utc),
                        updated_by=employee_id,
                    )
                )
        except Exception as e:
            await self.logger.error("Failed to touch session", e)
            raise database_error(str(e))
