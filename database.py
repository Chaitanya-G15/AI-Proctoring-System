import sqlite3
from datetime import datetime
import json

class Database:
    """
    SQLite database wrapper for the proctoring system.
    Handles students and violations storage.
    """

    def __init__(self, db_path='proctoring.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self.create_tables()

    def create_tables(self):
        """Create the database schema if it doesn't exist."""
        # Students table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                joined_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Violations table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                violation_type TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                screenshot_path TEXT,
                details TEXT,
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        ''')

        self.conn.commit()

    def add_student(self, name):
        """
        Add a new student or return existing student ID.

        Args:
            name: Student name

        Returns:
            int: Student ID
        """
        try:
            cursor = self.conn.execute('INSERT INTO students (name) VALUES (?)', (name,))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Student already exists, return their ID
            cursor = self.conn.execute('SELECT id FROM students WHERE name = ?', (name,))
            row = cursor.fetchone()
            if row:
                return row['id']
            raise

    def add_violation(self, student_id, violation_type, screenshot_path=None, details=None):
        """
        Add a violation for a student.

        Args:
            student_id: Student ID
            violation_type: Type of violation ('not_focused', 'phone', 'multiple_people')
            screenshot_path: Optional path to screenshot
            details: Optional dict with additional details (will be JSON encoded)
        """
        details_json = json.dumps(details) if details else None

        self.conn.execute(
            'INSERT INTO violations (student_id, violation_type, screenshot_path, details) VALUES (?, ?, ?, ?)',
            (student_id, violation_type, screenshot_path, details_json)
        )
        self.conn.commit()

    def update_student_heartbeat(self, student_id):
        """
        Update the last_seen timestamp for a student.

        Args:
            student_id: Student ID
        """
        self.conn.execute(
            'UPDATE students SET last_seen = ? WHERE id = ?',
            (datetime.now(), student_id)
        )
        self.conn.commit()

    def get_active_students(self, timeout_seconds=10):
        """
        Get all students who were seen within the last timeout_seconds.

        Args:
            timeout_seconds: How many seconds ago to consider "active"

        Returns:
            list: List of dicts with student info
        """
        cursor = self.conn.execute(
            '''SELECT id, name, joined_at, last_seen
               FROM students
               WHERE last_seen > datetime('now', ? || ' seconds')
               ORDER BY name''',
            (-timeout_seconds,)
        )

        students = []
        for row in cursor.fetchall():
            students.append({
                'id': row['id'],
                'name': row['name'],
                'joined_at': row['joined_at'],
                'last_seen': row['last_seen']
            })

        return students

    def get_all_students(self):
        """
        Get all students (active or inactive).

        Returns:
            list: List of dicts with student info
        """
        cursor = self.conn.execute(
            'SELECT id, name, joined_at, last_seen FROM students ORDER BY joined_at DESC'
        )

        students = []
        for row in cursor.fetchall():
            students.append({
                'id': row['id'],
                'name': row['name'],
                'joined_at': row['joined_at'],
                'last_seen': row['last_seen']
            })

        return students

    def get_violation_count(self, student_id):
        """
        Get total violation count for a student.

        Args:
            student_id: Student ID

        Returns:
            int: Total violations
        """
        cursor = self.conn.execute(
            'SELECT COUNT(*) as count FROM violations WHERE student_id = ?',
            (student_id,)
        )
        return cursor.fetchone()['count']

    def get_violations_by_type(self, student_id):
        """
        Get violation counts grouped by type.

        Args:
            student_id: Student ID

        Returns:
            dict: {violation_type: count}
        """
        cursor = self.conn.execute(
            '''SELECT violation_type, COUNT(*) as count
               FROM violations
               WHERE student_id = ?
               GROUP BY violation_type''',
            (student_id,)
        )

        violations = {}
        for row in cursor.fetchall():
            violations[row['violation_type']] = row['count']

        return violations

    def get_latest_screenshot(self, student_id):
        """
        Get the path to the latest screenshot for a student.

        Args:
            student_id: Student ID

        Returns:
            str or None: Screenshot path, or None if no screenshots
        """
        cursor = self.conn.execute(
            '''SELECT screenshot_path
               FROM violations
               WHERE student_id = ? AND screenshot_path IS NOT NULL
               ORDER BY timestamp DESC
               LIMIT 1''',
            (student_id,)
        )

        row = cursor.fetchone()
        return row['screenshot_path'] if row else None

    def get_student_by_id(self, student_id):
        """
        Get student info by ID.

        Args:
            student_id: Student ID

        Returns:
            dict or None: Student info
        """
        cursor = self.conn.execute(
            'SELECT id, name, joined_at, last_seen FROM students WHERE id = ?',
            (student_id,)
        )

        row = cursor.fetchone()
        if row:
            return {
                'id': row['id'],
                'name': row['name'],
                'joined_at': row['joined_at'],
                'last_seen': row['last_seen']
            }
        return None

    def get_violations(self, student_id, limit=None):
        """
        Get violations for a student.

        Args:
            student_id: Student ID
            limit: Optional limit on number of violations to return

        Returns:
            list: List of dicts with violation info
        """
        query = '''SELECT id, violation_type, timestamp, screenshot_path, details
                   FROM violations
                   WHERE student_id = ?
                   ORDER BY timestamp DESC'''

        if limit:
            query += f' LIMIT {limit}'

        cursor = self.conn.execute(query, (student_id,))

        violations = []
        for row in cursor.fetchall():
            violations.append({
                'id': row['id'],
                'violation_type': row['violation_type'],
                'timestamp': row['timestamp'],
                'screenshot_path': row['screenshot_path'],
                'details': json.loads(row['details']) if row['details'] else {}
            })

        return violations

    def close(self):
        """Close the database connection."""
        self.conn.close()
