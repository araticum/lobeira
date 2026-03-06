from __future__ import annotations

import io
import logging
import os
import posixpath
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ZipExtractionLimits:
    max_depth: int = int(os.getenv("PARSER_ZIP_MAX_DEPTH", "12"))
    max_files: int = int(os.getenv("PARSER_ZIP_MAX_FILES", "2000"))
    max_total_bytes: int = int(os.getenv("PARSER_ZIP_MAX_TOTAL_BYTES", str(500 * 1024 * 1024)))


@dataclass
class ZipExtractionResult:
    files: list[tuple[str, bytes]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    visited_archives: int = 0
    extracted_files: int = 0
    total_extracted_bytes: int = 0


def _is_zip_name(name: str) -> bool:
    return name.lower().endswith(".zip")


def _is_zip_bytes(content: bytes) -> bool:
    try:
        return zipfile.is_zipfile(io.BytesIO(content))
    except Exception:
        return False


def _sanitize_zip_member_name(name: str) -> str | None:
    raw = (name or "").replace("\\", "/")
    if not raw:
        return None

    normalized = posixpath.normpath(raw)
    if normalized in ("", "."):
        return None
    if normalized.startswith("../") or normalized == "..":
        return None
    if normalized.startswith("/"):
        return None
    if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/":
        return None

    return normalized


def extract_zip_recursive(
    *,
    archive_name: str,
    archive_bytes: bytes,
    limits: ZipExtractionLimits | None = None,
    logger: logging.Logger | None = None,
    log_prefix: str = "parser-zip",
) -> ZipExtractionResult:
    limits = limits or ZipExtractionLimits()
    logger = logger or logging.getLogger("zip-recursive")

    result = ZipExtractionResult()

    def _log_info(msg: str) -> None:
        logger.info("[%s] %s", log_prefix, msg)

    def _warn(msg: str) -> None:
        result.warnings.append(msg)
        logger.warning("[%s] %s", log_prefix, msg)

    _log_info(
        f"start archive={archive_name} max_depth={limits.max_depth} max_files={limits.max_files} max_total_bytes={limits.max_total_bytes}"
    )

    def _extract(current_name: str, current_bytes: bytes, depth: int, branch: str) -> None:
        if depth > limits.max_depth:
            _warn(f"skip depth-exceeded depth={depth} archive={current_name} branch={branch}")
            return

        try:
            with zipfile.ZipFile(io.BytesIO(current_bytes)) as zf:
                result.visited_archives += 1
                members = zf.infolist()
                _log_info(f"tree depth={depth} archive={current_name} entries={len(members)} branch={branch}")

                for info in members:
                    member_raw = info.filename
                    member_name = _sanitize_zip_member_name(member_raw)
                    branch_path = f"{branch}/{member_raw}" if branch else member_raw

                    if info.is_dir():
                        _log_info(f"tree depth={depth} dir={member_raw} branch={branch}")
                        continue

                    if not member_name:
                        _warn(f"skip dangerous-path member={member_raw} branch={branch}")
                        continue

                    if result.extracted_files >= limits.max_files:
                        _warn(f"skip max-files-reached member={member_name} branch={branch}")
                        continue

                    projected_total = result.total_extracted_bytes + max(0, int(info.file_size or 0))
                    if projected_total > limits.max_total_bytes:
                        _warn(
                            f"skip max-total-bytes member={member_name} file_size={info.file_size} total_if_extracted={projected_total}"
                        )
                        continue

                    try:
                        content = zf.read(info)
                    except Exception as exc:
                        _warn(f"skip read-error member={member_name} branch={branch} err={exc}")
                        continue

                    real_total = result.total_extracted_bytes + len(content)
                    if real_total > limits.max_total_bytes:
                        _warn(
                            f"skip max-total-bytes-after-read member={member_name} inflated={len(content)} total_if_extracted={real_total}"
                        )
                        continue

                    if _is_zip_name(member_name) or _is_zip_bytes(content):
                        _log_info(f"tree depth={depth} nested-zip={member_name} size={len(content)} branch={branch}")
                        _extract(member_name, content, depth + 1, branch_path)
                        continue

                    safe_branch = _sanitize_zip_member_name(branch)
                    output_name = f"{safe_branch}/{member_name}" if safe_branch else member_name
                    result.files.append((output_name, content))
                    result.extracted_files += 1
                    result.total_extracted_bytes = real_total
                    _log_info(
                        f"tree depth={depth} file={output_name} size={len(content)} extracted_files={result.extracted_files} total_bytes={result.total_extracted_bytes}"
                    )
        except zipfile.BadZipFile as exc:
            _warn(f"skip bad-zip archive={current_name} branch={branch} err={exc}")
        except Exception as exc:
            _warn(f"skip archive-error archive={current_name} branch={branch} err={exc}")

    _extract(archive_name, archive_bytes, 0, archive_name)
    _log_info(
        f"done archive={archive_name} visited_archives={result.visited_archives} extracted_files={result.extracted_files} total_bytes={result.total_extracted_bytes} warnings={len(result.warnings)}"
    )
    return result


def write_extracted_files(dest_dir: Path, extracted: list[tuple[str, bytes]], logger: logging.Logger | None = None) -> list[Path]:
    logger = logger or logging.getLogger("zip-recursive")
    saved: list[Path] = []

    for rel_name, content in extracted:
        rel_path = Path(rel_name)
        candidate = dest_dir / rel_path
        candidate.parent.mkdir(parents=True, exist_ok=True)

        if candidate.exists():
            stem = candidate.stem
            suffix = candidate.suffix
            parent = candidate.parent
            idx = 1
            while True:
                alt = parent / f"{stem}_{idx}{suffix}"
                if not alt.exists():
                    candidate = alt
                    break
                idx += 1

        candidate.write_bytes(content)
        saved.append(candidate)
        logger.info("[parser-zip] wrote file=%s bytes=%s", candidate, len(content))

    return saved
