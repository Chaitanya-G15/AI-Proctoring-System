"""SQLite database wrapper for proctoring system."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class Database:
    """Manages student and violation data in SQLite."""

    # Violation priority levels (higher = more severe)
    VIOLATION_PRIORITY = {
        'phone': 4,           # Highest priority
        'book': 3,
        'multiple_people': 2,
        'not_focused': 1      # Lowest priority
    }

    def __init__(self, db_path: str | Path = "proctoring.db") -> None:
        """Initialize database connection and create tables.

        Args:
            db_path: Path to SQLite database file
        """
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database schema if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                joined_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                violation_type TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                confidence REAL DEFAULT 1.0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                screenshot_path TEXT,
                details TEXT,
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        """)

        # Add priority column to existing tables (if they don't have it)
        try:
            self.conn.execute("ALTER TABLE violations ADD COLUMN priority INTEGER NOT NULL DEFAULT 0")
            self.conn.commit()
        except sqlite3.OperationalError:
            # Column already exists
            pass

        # Add confidence column to existing tables (if they don't have it)
        try:
            self.conn.execute("ALTER TABLE violations ADD COLUMN confidence REAL DEFAULT 1.0")
            self.conn.commit()
        except sqlite3.OperationalError:
            # Column already exists
            pass

        self.conn.commit()

    def add_student(self, name: str) -> int:
        """Add student or return existing student ID.

        Args:
            name: Student name

        Returns:
            Student ID
        """
        try:
            cursor = self.conn.execute("INSERT INTO students (name) VALUES (?)", (name,))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Student exists, return their ID
            cursor = self.conn.execute("SELECT id FROM students WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return row["id"]
            raise

    def add_violation(
        self,
        student_id: int,
        violation_type: str,
        screenshot_path: str | None = None,
        details: dict[str, Any] | None = None,
        confidence: float = 1.0,
    ) -> None:
        """Record a violation for a student.

        Args:
            student_id: Student ID
            violation_type: Type ('not_focused', 'phone', 'multiple_people', 'book')
            screenshot_path: Optional screenshot filename
            details: Optional violation metadata
            confidence: Detection confidence score (0.0-1.0, default 1.0)
        """
        details_json = json.dumps(details) if details else None
        priority = self.VIOLATION_PRIORITY.get(violation_type, 0)

        self.conn.execute(
            "INSERT INTO violations (student_id, violation_type, priority, confidence, screenshot_path, details) VALUES (?, ?, ?, ?, ?, ?)",
            (student_id, violation_type, priority, confidence, screenshot_path, details_json),
        )
        self.conn.commit()

    def update_student_heartbeat(self, student_id: int) -> None:
        """Update last_seen timestamp for a student.

        Args:
            student_id: Student ID
        """
        self.conn.execute(
            "UPDATE students SET last_seen = ? WHERE id = ?",
            (datetime.now(), student_id),
        )
        self.conn.commit()

    def get_active_students(self, timeout_seconds: int = 900) -> list[dict[str, Any]]:
        """Get students seen within timeout period.

        Args:
            timeout_seconds: How many seconds ago to consider "active" (default 15 minutes)

        Returns:
            List of student dicts with id, name, joined_at, last_seen
        """
        cursor = self.conn.execute(
            """SELECT id, name, joined_at, last_seen
               FROM students
               WHERE last_seen > datetime('now', ? || ' seconds')
               ORDER BY name""",
            (-timeout_seconds,),
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_all_students(self) -> list[dict[str, Any]]:
        """Get all students (active or inactive).

        Returns:
            List of student dicts ordered by join time (newest first)
        """
        cursor = self.conn.execute(
            "SELECT id, name, joined_at, last_seen FROM students ORDER BY joined_at DESC"
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_violation_count(self, student_id: int) -> int:
        """Get total violation count for a student.

        Args:
            student_id: Student ID

        Returns:
            Total violations
        """
        cursor = self.conn.execute(
            "SELECT COUNT(*) as count FROM violations WHERE student_id = ?",
            (student_id,),
        )
        return cursor.fetchone()["count"]

    def get_violations_by_type(self, student_id: int) -> dict[str, int]:
        """Get violation counts grouped by type.

        Args:
            student_id: Student ID

        Returns:
            Dict mapping violation_type to count
        """
        cursor = self.conn.execute(
            """SELECT violation_type, COUNT(*) as count
               FROM violations
               WHERE student_id = ?
               GROUP BY violation_type""",
            (student_id,),
        )

        return {row["violation_type"]: row["count"] for row in cursor.fetchall()}

    def get_latest_screenshot(self, student_id: int) -> str | None:
        """Get the highest priority violation screenshot for a student.

        Priority order: phone > book > multiple_people > not_focused
        Within same priority, returns highest confidence detection.

        Args:
            student_id: Student ID

        Returns:
            Screenshot filename or None if no screenshots exist
        """
        cursor = self.conn.execute(
            """SELECT screenshot_path
               FROM violations
               WHERE student_id = ? AND screenshot_path IS NOT NULL
               ORDER BY priority DESC, confidence DESC, timestamp DESC
               LIMIT 1""",
            (student_id,),
        )

        row = cursor.fetchone()
        return row["screenshot_path"] if row else None

    def get_highest_priority_violation(self, student_id: int) -> dict[str, Any] | None:
        """Get the highest priority violation for a student (with screenshot).

        Priority order: phone > book > multiple_people > not_focused
        Within same priority, returns highest confidence detection.

        Args:
            student_id: Student ID

        Returns:
            Dict with violation info or None if no violations exist
        """
        cursor = self.conn.execute(
            """SELECT violation_type, screenshot_path, timestamp, priority, confidence
               FROM violations
               WHERE student_id = ? AND screenshot_path IS NOT NULL
               ORDER BY priority DESC, confidence DESC, timestamp DESC
               LIMIT 1""",
            (student_id,),
        )

        row = cursor.fetchone()
        return dict(row) if row else None

    def get_student_by_id(self, student_id: int) -> dict[str, Any] | None:
        """Get student info by ID.

        Args:
            student_id: Student ID

        Returns:
            Student dict or None if not found
        """
        cursor = self.conn.execute(
            "SELECT id, name, joined_at, last_seen FROM students WHERE id = ?",
            (student_id,),
        )

        row = cursor.fetchone()
        return dict(row) if row else None

    def get_violations(
        self, student_id: int, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get violations for a student.

        Args:
            student_id: Student ID
            limit: Optional max number of violations to return

        Returns:
            List of violation dicts (newest first)
        """
        query = """SELECT id, violation_type, timestamp, screenshot_path, details
                   FROM violations
                   WHERE student_id = ?
                   ORDER BY timestamp DESC"""

        if limit:
            query += f" LIMIT {limit}"

        cursor = self.conn.execute(query, (student_id,))

        violations = []
        for row in cursor.fetchall():
            violations.append(
                {
                    "id": row["id"],
                    "violation_type": row["violation_type"],
                    "timestamp": row["timestamp"],
                    "screenshot_path": row["screenshot_path"],
                    "details": json.loads(row["details"]) if row["details"] else {},
                }
            )

        return violations

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
