"""Durable storage for completed zone visits.

export_db.py has always read a `zone_visits` table out of a SQLite database.
Nothing wrote one - dwell.py could append completed visits to a CSV and that
was all - so the export could never run against a real file. This is the
missing half: the same rows dwell.py already builds, written to the table
export_db.py already expects.

The column names are not a new invention. dwell.py's CSV header was

    entered_at, left_at, person_id, zone_id, seconds

and export_db.py reads exactly those, plus an `id` it uses only to break ties
between rows written in the same second. Both halves were written to the same
shape; only the sink between them was missing.

`camera_id` is the one addition. There is one camera today, so it is always
the same value - but every row a multi-camera setup will ever need to
distinguish is being written now, and adding the column later would mean
migrating a database the tracker is actively appending to. export_db.py picks
its columns up from PRAGMA table_info, so the extra field flows into the CSV
and the .xlsx raw sheet without that file changing at all.

Writes are best-effort in the same way dwell.py's CSV logging is: a storage
failure must never take the tracker down. Errors are reported once and then
suppressed, because a broken disk would otherwise print on every visit.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Optional

TABLE = "zone_visits"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id  TEXT    NOT NULL DEFAULT 'cam1',
    person_id  INTEGER NOT NULL,
    zone_id    TEXT    NOT NULL,
    entered_at TEXT    NOT NULL,
    left_at    TEXT    NOT NULL,
    seconds    REAL    NOT NULL
);
-- entered_at drives every time-based table in export_db.py, and person_id
-- drives the per-person journeys; both are worth an index once a busy day has
-- put a few hundred thousand rows in here.
CREATE INDEX IF NOT EXISTS idx_zone_visits_entered ON {TABLE} (entered_at);
CREATE INDEX IF NOT EXISTS idx_zone_visits_person  ON {TABLE} (person_id);
"""

_TS = "%Y-%m-%d %H:%M:%S"


def format_ts(epoch: float) -> str:
    """Epoch seconds -> the timestamp format export_db.parse_ts() reads.

    Local time, matching what dwell.py already writes to its CSV, so a database
    and a CSV from the same run agree.
    """
    return time.strftime(_TS, time.localtime(epoch))


class VisitStore:
    """Appends completed zone visits to a SQLite database.

    The connection is opened lazily, on the first write, because the pipeline
    builds this object on the main thread and then writes from its own thread.
    Opening where we write keeps the handle and the thread together, and means
    a bad --db path fails when it is used rather than at import time.
    """

    def __init__(self, db_path: str, camera_id: str = "cam1") -> None:
        self.db_path = db_path
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        # The database could not be opened at all - there is nothing to retry,
        # so stop trying.
        self._failed = False
        # A write failed at least once. Writes keep being attempted (a locked
        # database usually frees up), but only the first failure is printed.
        self._reported = False
        self._rows = 0

    # ------------------------------------------------------------------
    def _connect(self) -> Optional[sqlite3.Connection]:
        if self._conn is not None or self._failed:
            return self._conn
        try:
            parent = os.path.dirname(os.path.abspath(self.db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            # check_same_thread=False plus our own lock: a later multi-camera
            # build will have several pipeline threads sharing one store, and
            # SQLite's own thread check would reject that before our lock ever
            # got a chance to serialise them.
            conn = sqlite3.connect(self.db_path, check_same_thread=False,
                                   timeout=5.0)
            # WAL is what makes export_db.py's "safe to run while the tracker
            # is writing" claim actually true: readers no longer block on the
            # writer, and the writer no longer blocks on them.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(SCHEMA)
            conn.commit()
            self._conn = conn
        except sqlite3.Error as exc:
            self._failed = True
            print(f"  [store] could not open {self.db_path}: {exc}")
            print("  [store] zone visits will not be recorded this run.")
        except OSError as exc:
            self._failed = True
            print(f"  [store] could not create the folder for {self.db_path}: {exc}")
        return self._conn

    # ------------------------------------------------------------------
    def add_visit(self, person_id: int, zone_id: str, since: float,
                  until: float, duration: float) -> None:
        """Record one completed visit. Never raises."""
        with self._lock:
            conn = self._connect()
            if conn is None:
                return
            try:
                conn.execute(
                    f"INSERT INTO {TABLE} "
                    "(camera_id, person_id, zone_id, entered_at, left_at, seconds) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (self.camera_id, int(person_id), str(zone_id),
                     format_ts(since), format_ts(until), round(float(duration), 1)),
                )
                conn.commit()
                self._rows += 1
            except sqlite3.Error as exc:
                # Report once, then keep trying quietly. A locked database
                # normally frees up a moment later, so refusing to write again
                # would throw away a whole session over one transient error -
                # but printing on every visit for the rest of the run would
                # bury the console.
                if not self._reported:
                    self._reported = True
                    print(f"  [store] a visit could not be written: {exc}")
                    print("  [store] further write errors will not be reported.")

    # ------------------------------------------------------------------
    @property
    def rows_written(self) -> int:
        return self._rows

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None
