"""Encrypted, access-logged storage for sensitive sourcehunt artifacts.

Exploits, PoCs, and transcripts are AES-256-GCM encrypted at rest with an
auto-generated key. Every store/retrieve operation is logged to an append-only
JSONL audit trail.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ArtifactPolicy:
    encryption_at_rest: bool = True
    access_logged: bool = True
    retention_days: int = 180
    tied_to_disclosure: bool = True
    export_requires_approval: bool = True


def _default_base_dir() -> Path:
    from clearwing.core.config import clearwing_home

    return clearwing_home() / "sourcehunt" / "artifacts"


class ArtifactStore:
    """Encrypted, access-logged storage for sensitive artifacts."""

    def __init__(
        self,
        base_dir: Path | None = None,
        policy: ArtifactPolicy | None = None,
        audit_logger: object | None = None,
    ):
        self._base_dir = Path(base_dir) if base_dir else _default_base_dir()
        self._policy = policy or ArtifactPolicy()
        self._audit = audit_logger
        self._key: bytes = b""
        self._init_storage()

    def _init_storage(self) -> None:
        for subdir in ("exploits", "transcripts", "poc", "keys"):
            (self._base_dir / subdir).mkdir(parents=True, exist_ok=True)

        key_path = self._base_dir / "keys" / "master.key"
        if key_path.exists():
            self._key = key_path.read_bytes()
        else:
            self._key = os.urandom(32)
            key_path.write_bytes(self._key)
            os.chmod(key_path, 0o600)

    def _encrypt(self, data: bytes) -> bytes:
        if not self._policy.encryption_at_rest:
            return data
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = os.urandom(12)
        ct = AESGCM(self._key).encrypt(nonce, data, None)
        return nonce + ct

    def _decrypt(self, blob: bytes) -> bytes:
        if not self._policy.encryption_at_rest:
            return blob
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce, ct = blob[:12], blob[12:]
        return AESGCM(self._key).decrypt(nonce, ct, None)

    def _log_access(self, action: str, finding_id: str, path: str, operator: str) -> None:
        if not self._policy.access_logged:
            return
        entry = {
            "timestamp": time.time(),
            "action": action,
            "finding_id": finding_id,
            "path": path,
            "operator": operator,
        }
        audit_path = self._base_dir / "audit.log"
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _store(self, category: str, finding_id: str, data: bytes, operator: str) -> Path:
        safe_id = finding_id.replace("/", "_").replace("..", "_")
        out_path = self._base_dir / category / f"{safe_id}.enc"
        out_path.write_bytes(self._encrypt(data))
        self._log_access(f"store_{category}", finding_id, str(out_path), operator)
        return out_path

    def store_exploit(self, finding_id: str, data: bytes, operator: str = "system") -> Path:
        return self._store("exploits", finding_id, data, operator)

    def store_poc(self, finding_id: str, data: bytes, operator: str = "system") -> Path:
        return self._store("poc", finding_id, data, operator)

    def store_transcript(self, finding_id: str, data: bytes, operator: str = "system") -> Path:
        return self._store("transcripts", finding_id, data, operator)

    def retrieve(self, artifact_path: Path, operator: str = "system") -> bytes:
        artifact_path = Path(artifact_path)
        data = self._decrypt(artifact_path.read_bytes())
        finding_id = artifact_path.stem
        self._log_access("retrieve", finding_id, str(artifact_path), operator)
        return data

    def list_artifacts(self, finding_id: str) -> list[dict]:
        safe_id = finding_id.replace("/", "_").replace("..", "_")
        results = []
        for category in ("exploits", "transcripts", "poc"):
            p = self._base_dir / category / f"{safe_id}.enc"
            if p.exists():
                stat = p.stat()
                results.append({
                    "category": category,
                    "path": str(p),
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                })
        return results

    def purge_expired(self) -> int:
        cutoff = time.time() - (self._policy.retention_days * 86400)
        removed = 0
        for category in ("exploits", "transcripts", "poc"):
            cat_dir = self._base_dir / category
            if not cat_dir.exists():
                continue
            for f in cat_dir.iterdir():
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
        return removed
