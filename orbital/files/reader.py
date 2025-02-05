import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileReader:
    @classmethod
    def _safe_read_json(
        cls, filename: str, snapshots_dir: Path | None = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
        """Read JSON file with pattern matching."""
        try:
            # If filename is an absolute path, check if it exists directly
            file_path = Path(filename)
            if file_path.is_absolute():
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        return json.load(f), file_path
                return None, None

            # Handle relative patterns
            if snapshots_dir is None:
                raise ValueError("found relative pattern but snapshot_dir is empty")

            patterns = [
                filename,
                f"*-of-*.{filename}",
                f"*/{filename}",
                f"**/{filename}",
            ]

            for pattern in patterns:
                matches = list(snapshots_dir.rglob(pattern))
                if matches:
                    file_path = matches[0]
                    with open(file_path, "r", encoding="utf-8") as f:
                        return json.load(f), file_path

            logger.debug(f"{filename} not found")
            return None, None

        except Exception as e:
            logger.warning(f"Error reading {filename}: {str(e)}")
            return None, None

    @classmethod
    def _safe_read_text(
        cls, filename: str, snapshots_dir: Path | None = None
    ) -> Tuple[Optional[List[str]], Optional[Path]]:
        try:
            # If filename is an absolute path, check if it exists directly
            file_path = Path(filename)
            if file_path.is_absolute():
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        return [line.strip() for line in f if line.strip()], file_path
                return None, None

            # Handle relative patterns
            if snapshots_dir is None:
                raise ValueError("found relative pattern but snapshot_dir is empty")

            file_path = next(snapshots_dir.rglob(filename), None)
            if not file_path:
                return None, None

            with open(file_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()], file_path

        except Exception as e:
            logger.warning(f"Error reading {filename}: {str(e)}")
            return None, None
