import sys
from contextlib import contextmanager
from typing import TextIO


class TeeStream:
    def __init__(self, stream: TextIO, log_file: TextIO):
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    @staticmethod
    @contextmanager
    def capture_learning_output(learning_log_path: str):
        # Keep console behavior while persisting all learning prints to a text log.
        with open(learning_log_path, "a", encoding="utf-8") as log_file:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = TeeStream(original_stdout, log_file)
            sys.stderr = TeeStream(original_stderr, log_file)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                sys.stdout = original_stdout
                sys.stderr = original_stderr
