from sqlalchemy import select

from repositories.Database import Database
from repositories.schemas.schema import Escalation
from utils.Exceptions.errorcodes import database_error
from utils.logger import Logger


class EscalationRepository:
    """All database operations for the escalations table."""

    def __init__(self) -> None:
        self.db = Database()
        self.logger = Logger("escalation_repository")

    # ── Create ─────────────────────────────────────────────────────────────────
    async def create_escalation(
        self,
        employee_id: str,
        session_id: str,
        message: str,
        reason: str,
        reasoning: str | None = None,
    ) -> Escalation:
        """Insert a new escalation record and return it."""
        try:
            async with self.db.get_session() as db_session:
                row = Escalation(
                    employee_id=employee_id,
                    session_id=session_id,
                    message=message,
                    reason=reason,
                    reasoning=reasoning,
                    status="open",
                    created_by=employee_id,
                    updated_by=employee_id,
                )
                db_session.add(row)
                await db_session.flush()
                self.logger.info(
                    "Escalation created",
                    employee_id=employee_id,
                    session_id=session_id,
                    reason=reason,
                )
                return row
        except Exception as e:
            await self.logger.error("Failed to create escalation", e)
            raise database_error(str(e))

    # ── Read ───────────────────────────────────────────────────────────────────
    async def get_escalations_by_employee(self, employee_id: str) -> list[Escalation]:
        """Return all escalations for a given employee, newest first."""
        try:
            async with self.db.get_session() as db_session:
                result = await db_session.execute(
                    select(Escalation)
                    .where(Escalation.employee_id == employee_id)
                    .order_by(Escalation.created_at.desc())
                )
                return list(result.scalars().all())
        except Exception as e:
            await self.logger.error("Failed to fetch escalations", e)
            raise database_error(str(e))

    async def get_escalations_by_session(self, session_id: str) -> list[Escalation]:
        """Return all escalations for a given session."""
        try:
            async with self.db.get_session() as db_session:
                result = await db_session.execute(
                    select(Escalation)
                    .where(Escalation.session_id == session_id)
                    .order_by(Escalation.created_at.desc())
                )
                return list(result.scalars().all())
        except Exception as e:
            await self.logger.error("Failed to fetch session escalations", e)
            raise database_error(str(e))

    # ── Update status ──────────────────────────────────────────────────────────
    async def close_escalation(self, escalation_id: int, updated_by: str) -> None:
        """Mark an escalation as closed."""
        try:
            async with self.db.get_session() as db_session:
                result = await db_session.get(Escalation, escalation_id)
                if result:
                    result.status = "closed"
                    result.updated_by = updated_by
        except Exception as e:
            await self.logger.error("Failed to close escalation", e)
            raise database_error(str(e))
