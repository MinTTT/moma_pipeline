"""Standard-library-only transactions for replacing related files.

Individual replacements are atomic, but a group of replacements is not. A
journal and content-verified backups make an interrupted group recoverable.
Call recover_transaction before using a directory whose journal may be
incomplete. Recovery is conservative: an unexpected destination hash stops
the whole recovery before any destination is changed.

The persistent <journal>.lock file is locked at the OS level, so the lock is
released when its owner exits (including a crash); the file itself is retained.
Only files and temporary paths explicitly recorded by this module are removed.

Ported verbatim from Moma_cellPose.moma_file_transactions into
moma_pipeline.io.transactions (src-layout package port).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any
import uuid


_JOURNAL_VERSION = 1
_INCOMPLETE_STATES = frozenset(("preparing", "committing"))
_FINISHED_STATES = frozenset(("complete", "rolled_back"))
_STATES = _INCOMPLETE_STATES | _FINISHED_STATES
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_READ_SIZE = 1024 * 1024


def sha256(path: Path | str) -> str:
    """Return the lowercase SHA-256 digest of a regular file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(_READ_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute(path: Path | str) -> Path:
    # abspath preserves a final symlink rather than silently retargeting it.
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _unlink_missing(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _fsync_directory(directory: Path) -> None:
    """Persist a directory entry where directory fsync is available."""
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _journal_lock(journal_path: Path) -> Iterator[None]:
    """Hold an exclusive, crash-released, per-journal interprocess lock."""
    lock_path = journal_path.with_name(journal_path.name + ".lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with os.fdopen(descriptor, "r+b", buffering=0) as stream:
            descriptor = -1
            # Windows byte-range locks require the locked byte to exist.
            # Do not read that byte first: another owner's lock can deny reads.
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise FileExistsError(
                    f"Transaction journal is in use: {journal_path}"
                ) from error
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                # Closing also releases the lock if explicit unlock fails.
    finally:
        if descriptor >= 0:
            os.close(descriptor)


# A stable per-FOV caller lock can span recovery, computation and publication.
transaction_lock = _journal_lock


def _new_temporary(
    directory: Path, prefix: str, suffix: str = ".tmp"
) -> tuple[int, Path]:
    descriptor, name = tempfile.mkstemp(
        dir=directory, prefix=prefix, suffix=suffix
    )
    return descriptor, Path(name)


def atomic_json(path: Path | str, data: Any) -> None:
    """Durably replace path with strict JSON (NaN and infinity rejected)."""
    destination = _absolute(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            data,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = _new_temporary(
        destination.parent, f".{destination.name}.json."
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        _unlink_missing(temporary)


def _validate_staged_file(path: Path, suffix: str) -> None:
    """Reload staged production formats without importing third-party code."""
    suffix = suffix.casefold()
    if suffix == ".pkl":
        with path.open("rb") as stream:
            pickle.load(stream)
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for _row in csv.reader(stream, strict=True):
                pass


def _stage_payload(
    destination: Path, payload: bytes, transaction_id: str
) -> tuple[Path, str]:
    if type(payload) is not bytes:
        raise TypeError(f"Payload for {destination} must be bytes")
    descriptor, staged = _new_temporary(
        destination.parent,
        f".{destination.name}.{transaction_id}.",
        destination.suffix or ".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256(staged)
        if digest != hashlib.sha256(payload).hexdigest():
            raise OSError(f"Staged content hash mismatch for {destination}")
        _validate_staged_file(staged, destination.suffix)
        _fsync_directory(staged.parent)
    except BaseException:
        _unlink_missing(staged)
        raise
    return staged, digest


def _stage_copy(
    source: Path, directory: Path, prefix: str, expected_sha256: str
) -> Path:
    descriptor, temporary = _new_temporary(directory, prefix)
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output:
            while True:
                block = input_stream.read(_READ_SIZE)
                if not block:
                    break
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        actual = sha256(temporary)
        if actual != expected_sha256:
            raise OSError(
                f"Content changed while copying {source}: expected "
                f"{expected_sha256}, found {actual}"
            )
        _fsync_directory(directory)
    except BaseException:
        _unlink_missing(temporary)
        raise
    return temporary


def _require_regular_destination(path: Path) -> bool:
    if path.is_symlink():
        raise ValueError(f"Symbolic-link destinations are not supported: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"Destination is not a regular file: {path}")
    return path.exists()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant in transaction journal: {value}")


def _read_journal(journal_path: Path) -> dict[str, Any]:
    with journal_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream, parse_constant=_reject_json_constant)
    _validate_journal(document)
    return document


def _validate_journal(document: Any) -> None:
    """Fail closed on malformed journals before touching any recorded path."""
    if type(document) is not dict:
        raise ValueError("Transaction journal must be a JSON object")
    if type(document.get("version")) is not int or document["version"] != 1:
        raise ValueError("Unsupported transaction journal version")
    transaction_id = document.get("transaction_id")
    if type(transaction_id) is not str or not transaction_id:
        raise ValueError("Journal transaction_id must be a nonempty string")
    state = document.get("state")
    if type(state) is not str or state not in _STATES:
        raise ValueError("Invalid transaction journal state")
    if type(document.get("status")) is not str or document["status"] != state:
        raise ValueError("Journal status must be a string matching state")
    if type(document.get("created_at")) is not str:
        raise ValueError("Journal created_at must be a string")
    backup_directory = document.get("backup_dir")
    if backup_directory is not None:
        if type(backup_directory) is not str or not os.path.isabs(backup_directory):
            raise ValueError("Journal backup_dir must be null or an absolute path")
        if Path(backup_directory).name != transaction_id:
            raise ValueError("Journal backup directory does not match transaction_id")
    entries = document.get("files")
    if type(entries) is not list or not entries:
        raise ValueError("Journal files must be a nonempty array")

    destinations: set[str] = set()
    for index, entry in enumerate(entries):
        if type(entry) is not dict:
            raise ValueError("Journal file entries must be objects")
        for field in ("destination", "staged_path"):
            value = entry.get(field)
            if type(value) is not str or not os.path.isabs(value):
                raise ValueError(f"Journal {field} must be an absolute path string")
        destination = Path(entry["destination"])
        key = _path_key(destination)
        if key in destinations:
            raise ValueError("Duplicate destination in transaction journal")
        destinations.add(key)
        old_exists = entry.get("old_exists")
        if type(old_exists) is not bool:
            raise ValueError("Journal old_exists must be a Boolean")
        old_hash = entry.get("old_sha256")
        if old_exists and not _is_digest(old_hash):
            raise ValueError("Existing destination lacks its old SHA-256")
        if not old_exists and old_hash is not None:
            raise ValueError("New destination must have a null old_sha256")
        if not _is_digest(entry.get("staged_sha256")):
            raise ValueError("Invalid staged SHA-256")
        expected = entry.get("expected_source_sha256")
        if expected is not None and not _is_digest(expected):
            raise ValueError("Invalid expected source SHA-256")

        staged = Path(entry["staged_path"])
        staged_prefix = f".{destination.name}.{transaction_id}."
        staged_suffix = destination.suffix or ".tmp"
        if (
            staged.parent != destination.parent
            or not staged.name.startswith(staged_prefix)
            or not staged.name.endswith(staged_suffix)
        ):
            raise ValueError("Journal contains a foreign staged path")

        backup_path = entry.get("backup_path")
        backup_hash = entry.get("backup_sha256")
        if old_exists:
            if type(backup_directory) is not str:
                raise ValueError("Existing destination lacks a backup directory")
            if type(backup_path) is not str or not os.path.isabs(backup_path):
                raise ValueError("Existing destination lacks a backup path")
            backup = Path(backup_path)
            try:
                relative_backup = backup.relative_to(Path(backup_directory))
            except ValueError as error:
                raise ValueError("Journal contains a foreign backup path") from error
            if (
                not relative_backup.parts
                or ".." in relative_backup.parts
                or backup.name != destination.name
            ):
                raise ValueError("Journal contains a foreign backup path")
            if not _is_digest(backup_hash) or backup_hash != old_hash:
                raise ValueError("Backup and original SHA-256 values disagree")
        elif backup_path is not None or backup_hash is not None:
            raise ValueError("New destination must not have backup fields")


def _write_state(journal_path: Path, document: dict[str, Any], state: str) -> None:
    if state not in _STATES:
        raise ValueError(f"Invalid transaction state: {state}")
    document["state"] = state
    document["status"] = state
    _validate_journal(document)
    atomic_json(journal_path, document)


def _current_hash(path: Path) -> str | None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"Destination is no longer a regular file: {path}")
    return sha256(path) if path.exists() else None


def _remove_new_output(destination: Path, transaction_id: str) -> None:
    """Atomically detach a newly published output before deleting our copy."""
    descriptor, detached = _new_temporary(
        destination.parent, f".{destination.name}.{transaction_id}.remove."
    )
    os.close(descriptor)
    try:
        os.replace(destination, detached)
        _fsync_directory(destination.parent)
        detached.unlink()
        _fsync_directory(destination.parent)
    finally:
        _unlink_missing(detached)


def _rollback(document: dict[str, Any]) -> None:
    """Preflight all entries, then restore originals or remove new outputs."""
    transaction_id = document["transaction_id"]
    entries = document["files"]

    # No writes occur until every destination and every backup passes inspection.
    current_hashes: list[str | None] = []
    for entry in entries:
        if entry["old_exists"]:
            backup = Path(entry["backup_path"])
            if not backup.is_file() or sha256(backup) != entry["backup_sha256"]:
                raise RuntimeError(f"Backup is missing or modified: {backup}")
        destination = Path(entry["destination"])
        current = _current_hash(destination)
        allowed = {entry["old_sha256"], entry["staged_sha256"]}
        if current not in allowed:
            raise RuntimeError(
                "Refusing recovery because a destination changed independently: "
                f"{destination}; current SHA-256={current!r}"
            )
        current_hashes.append(current)

    for entry, current in reversed(list(zip(entries, current_hashes))):
        destination = Path(entry["destination"])
        if entry["old_exists"]:
            if current == entry["old_sha256"]:
                continue
            temporary = _stage_copy(
                Path(entry["backup_path"]),
                destination.parent,
                f".{destination.name}.{transaction_id}.restore.",
                entry["backup_sha256"],
            )
            try:
                # Re-check immediately before replacing an editable path.
                if _current_hash(destination) != current:
                    raise RuntimeError(
                        f"Destination changed during recovery: {destination}"
                    )
                os.replace(temporary, destination)
                _fsync_directory(destination.parent)
                if sha256(destination) != entry["old_sha256"]:
                    raise RuntimeError(f"Failed to restore original: {destination}")
            finally:
                _unlink_missing(temporary)
        elif current == entry["staged_sha256"]:
            if _current_hash(destination) != current:
                raise RuntimeError(f"Destination changed during recovery: {destination}")
            _remove_new_output(destination, transaction_id)


def _cleanup_staged_paths(paths: list[Path]) -> None:
    # These paths came from our own mkstemp calls, never from external input.
    for path in paths:
        _unlink_missing(path)


def recover_transaction(journal_path: Path | str) -> str:
    """Recover an incomplete transaction and return its resulting state.

    Returns "missing" when no journal exists, or "complete" / "rolled_back"
    unchanged for an already finished journal. Incomplete "preparing" and
    "committing" journals are restored and returned as "rolled_back".
    Modified destinations, bad backups, malformed journals, and active writers
    raise without being treated as successful recovery.
    """
    journal = _absolute(journal_path)
    if not journal.parent.is_dir() or not journal.exists():
        return "missing"
    with _journal_lock(journal):
        if not journal.exists():
            return "missing"
        document = _read_journal(journal)
        state = document["state"]
        if state in _FINISHED_STATES:
            return state
        _rollback(document)
        _write_state(journal, document, "rolled_back")
        _cleanup_staged_paths(
            [Path(entry["staged_path"]) for entry in document["files"]]
        )
        return "rolled_back"


def _normalise_expected_sources(
    expected_sources: Mapping[Path, str] | None,
    destinations: Mapping[Path, bytes],
) -> dict[str, str]:
    if expected_sources is None:
        return {}
    known = {_path_key(path) for path in destinations}
    result: dict[str, str] = {}
    for source, digest in expected_sources.items():
        path = _absolute(source)
        key = _path_key(path)
        if key not in known:
            raise ValueError(
                f"Expected source is not a publication destination: {path}"
            )
        if key in result:
            raise ValueError(f"Duplicate expected source: {path}")
        if not _is_digest(digest):
            raise ValueError(
                f"Expected source SHA-256 must be lowercase hexadecimal: {path}"
            )
        result[key] = digest
    return result


def publish_files(
    files: Mapping[Path, bytes],
    journal_path: Path,
    expected_sources: Mapping[Path, str] | None = None,
    backup_root: Path | None = None,
) -> dict[str, Any]:
    """Stage, validate, and publish a related group of byte payloads.

    expected_sources may cover all or a subset of destinations. Every original
    is snapshotted and rehashed before commit; supplied expectations are an
    additional guard. backup_root is a parent directory to which a unique
    transaction UUID directory is appended. Its default is
    <common-target-directory>/.cell_parameter_backups.

    On success the returned JSON-compatible dictionary has status,
    transaction_id, journal_path, backup_dir (string or null), and files (an
    absolute-destination to new-SHA-256 object). Existing journals are never
    reused, preserving finished audit history. Replacement errors trigger
    rollback and are re-raised only after restoration succeeds.
    """
    if not isinstance(files, Mapping) or not files:
        raise ValueError("files must be a nonempty mapping of destinations to bytes")
    journal = _absolute(journal_path)
    normalised_files: dict[Path, bytes] = {}
    keys: set[str] = set()
    for destination, payload in files.items():
        path = _absolute(destination)
        key = _path_key(path)
        if key in keys:
            raise ValueError(f"Duplicate publication destination: {path}")
        protected = {
            _path_key(journal),
            _path_key(journal.with_name(journal.name + ".lock")),
        }
        if key in protected:
            raise ValueError(
                "Journal and lock paths cannot be publication destinations"
            )
        keys.add(key)
        normalised_files[path] = payload
    expected = _normalise_expected_sources(expected_sources, normalised_files)

    journal.parent.mkdir(parents=True, exist_ok=True)
    for destination in normalised_files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _require_regular_destination(destination)

    with _journal_lock(journal):
        if journal.exists():
            previous = _read_journal(journal)
            if previous["state"] in _INCOMPLETE_STATES:
                detail = "recover the incomplete transaction before publication"
            else:
                detail = "finished journals are retained as audit history"
            raise FileExistsError(
                f"Transaction journal already exists ({detail}); "
                f"choose a new path: {journal}"
            )
        transaction_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        entries: list[dict[str, Any]] = []
        staged_paths: list[Path] = []
        try:
            # Stage every payload before recording originals or replacing anything.
            for destination, payload in normalised_files.items():
                staged, staged_hash = _stage_payload(
                    destination, payload, transaction_id
                )
                staged_paths.append(staged)
                entries.append(
                    {
                        "destination": str(destination),
                        "old_exists": False,
                        "old_sha256": None,
                        "backup_path": None,
                        "backup_sha256": None,
                        "staged_path": str(staged),
                        "staged_sha256": staged_hash,
                        "expected_source_sha256": expected.get(
                            _path_key(destination)
                        ),
                    }
                )

            old_hashes: dict[str, str] = {}
            for entry in entries:
                destination = Path(entry["destination"])
                if _require_regular_destination(destination):
                    old_hash = sha256(destination)
                    entry["old_exists"] = True
                    entry["old_sha256"] = old_hash
                    old_hashes[_path_key(destination)] = old_hash

            if old_hashes:
                common_directory = Path(
                    os.path.commonpath(
                        [str(path.parent) for path in normalised_files]
                    )
                )
                if backup_root is None:
                    root = common_directory / ".cell_parameter_backups"
                else:
                    root = _absolute(backup_root)
                backup_directory = root / transaction_id
                backup_directory.mkdir(parents=True, exist_ok=False)
                _fsync_directory(root)
                for entry in entries:
                    if not entry["old_exists"]:
                        continue
                    destination = Path(entry["destination"])
                    relative_parent = destination.parent.relative_to(common_directory)
                    backup = backup_directory / relative_parent / destination.name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    temporary = _stage_copy(
                        destination,
                        backup.parent,
                        f".{transaction_id}.backup.",
                        entry["old_sha256"],
                    )
                    try:
                        os.replace(temporary, backup)
                        _fsync_directory(backup.parent)
                    finally:
                        _unlink_missing(temporary)
                    entry["backup_path"] = str(backup)
                    entry["backup_sha256"] = sha256(backup)
                document_backup = str(backup_directory)
            else:
                document_backup = None

            document = {
                "version": _JOURNAL_VERSION,
                "transaction_id": transaction_id,
                "created_at": created_at,
                "state": "preparing",
                "status": "preparing",
                "backup_dir": document_backup,
                "files": entries,
            }
            _write_state(journal, document, "preparing")

            # This is deliberately the last source read before the commit point.
            for entry in entries:
                destination = Path(entry["destination"])
                current = _current_hash(destination)
                expected_hash = entry["expected_source_sha256"]
                if expected_hash is not None and current != expected_hash:
                    raise RuntimeError(
                        f"Source SHA-256 mismatch for {destination}: "
                        f"expected {expected_hash}, found {current}"
                    )
                if current != entry["old_sha256"]:
                    raise RuntimeError(
                        f"Source changed during publication: {destination}"
                    )

            _write_state(journal, document, "committing")
            for entry in entries:
                destination = Path(entry["destination"])
                os.replace(Path(entry["staged_path"]), destination)
                _fsync_directory(destination.parent)
            for entry in entries:
                actual = sha256(entry["destination"])
                if actual != entry["staged_sha256"]:
                    raise RuntimeError(
                        f"Published SHA-256 mismatch: {entry['destination']}"
                    )
            _write_state(journal, document, "complete")
            return {
                "status": "complete",
                "transaction_id": transaction_id,
                "journal_path": str(journal),
                "backup_dir": document["backup_dir"],
                "files": {
                    entry["destination"]: entry["staged_sha256"]
                    for entry in entries
                },
            }
        except BaseException:
            if journal.exists():
                try:
                    saved = _read_journal(journal)
                    if saved["state"] in _INCOMPLETE_STATES:
                        _rollback(saved)
                        _write_state(journal, saved, "rolled_back")
                except BaseException as recovery_error:
                    raise RuntimeError(
                        "Publication failed and automatic rollback did not "
                        f"complete; run recover_transaction({str(journal)!r}) "
                        "after resolving the reported conflict"
                    ) from recovery_error
            # Preserve the original replacement/validation exception.
            raise
        finally:
            _cleanup_staged_paths(staged_paths)


__all__ = ["atomic_json", "publish_files", "recover_transaction", "sha256"]
