import contextlib
import fcntl
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional


def _now_us() -> int:
    return int(time.time() * 1_000_000)


class PerfettoTrace:
    def __init__(self, events_path: Optional[str], json_path: Optional[str] = None, process_name: Optional[str] = None):
        self.enabled = bool(events_path)
        self.process_name = process_name or os.path.basename(os.getenv("PYTHON_EXECUTABLE", "python"))
        self._metadata_written = False
        self._disable_notice_emitted = False
        if not self.enabled:
            self.events_path = None
            self.json_path = None
            self.lock_path = None
            return

        self.events_path = Path(events_path)
        self.json_path = Path(json_path) if json_path else self.events_path.with_suffix(".json")
        self.lock_path = self.events_path.with_suffix(self.events_path.suffix + ".lock")
        if not self._ensure_output_paths():
            return

    def _disable(self, reason: str, exc: Optional[OSError] = None):
        self.enabled = False
        if self._disable_notice_emitted:
            return
        self._disable_notice_emitted = True
        message = f"[perfetto] tracing disabled: {reason}"
        if exc is not None:
            message += f" ({exc})"
        print(message, file=sys.stderr, flush=True)

    def _ensure_output_paths(self) -> bool:
        if not self.enabled:
            return False
        try:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._disable(f"unable to create trace output directory for {self.events_path}", exc)
            return False
        return True

    def _emit_metadata_if_needed(self):
        if self._metadata_written or not self.enabled:
            return
        self._metadata_written = True
        self._append_event(
            {
                "name": "process_name",
                "ph": "M",
                "pid": os.getpid(),
                "tid": threading.get_ident(),
                "ts": _now_us(),
                "args": {"name": self.process_name},
            }
        )

    def _append_event(self, event: dict):
        if not self.enabled:
            return
        if not self._ensure_output_paths():
            return
        try:
            with self.lock_path.open("a+", encoding="utf-8") as lock_f:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                with self.events_path.open("a", encoding="utf-8") as events_f:
                    events_f.write(json.dumps(event, ensure_ascii=False) + "\n")

                events = []
                with self.events_path.open("r", encoding="utf-8") as events_f:
                    for line in events_f:
                        line = line.strip()
                        if not line:
                            continue
                        events.append(json.loads(line))

                payload = {
                    "traceEvents": events,
                    "displayTimeUnit": "ms",
                }
                with self.json_path.open("w", encoding="utf-8") as json_f:
                    json.dump(payload, json_f, ensure_ascii=False)
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            self._disable(f"unable to write trace event to {self.events_path}", exc)

    def instant(self, name: str, cat: str, args: Optional[dict] = None):
        if not self.enabled:
            return
        self._emit_metadata_if_needed()
        self._append_event(
            {
                "name": name,
                "cat": cat,
                "ph": "i",
                "s": "t",
                "pid": os.getpid(),
                "tid": threading.get_ident(),
                "ts": _now_us(),
                "args": args or {},
            }
        )

    @contextlib.contextmanager
    def span(self, name: str, cat: str, args: Optional[dict] = None):
        if not self.enabled:
            yield
            return
        self._emit_metadata_if_needed()
        pid = os.getpid()
        tid = threading.get_ident()
        self._append_event(
            {
                "name": name,
                "cat": cat,
                "ph": "B",
                "pid": pid,
                "tid": tid,
                "ts": _now_us(),
                "args": args or {},
            }
        )
        try:
            yield
        finally:
            self._append_event(
                {
                    "name": name,
                    "cat": cat,
                    "ph": "E",
                    "pid": pid,
                    "tid": tid,
                    "ts": _now_us(),
                }
            )
