from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from app.schemas.wage import MinWageDataset, MinWageRecord

logger = logging.getLogger(__name__)

_CACHED: MinWageDataset | None = None


def _default_data_path() -> Path:
    # repo_root/resources/min_wage_hourly.json
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "resources" / "min_wage_hourly.json"


def _resolve_data_path() -> Path:
    env_path = (os.environ.get("MIN_WAGE_DATA_PATH") or "").strip()
    if env_path:
        p = Path(env_path)
        if p.is_absolute():
            return p
        repo_root = Path(__file__).resolve().parents[2]
        return (repo_root / p).resolve()
    return _default_data_path()


def load_min_wage_dataset() -> MinWageDataset:
    """
    Load hourly minimum wage dataset once and cache it in memory.

    - On missing/invalid file: returns empty dict and logs a warning.
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    path = _resolve_data_path()
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("dataset must be a JSON object")

        parsed: MinWageDataset = {}
        for k, v in data.items():
            cc = str(k or "").strip().upper()
            if len(cc) != 2 or not cc.isalpha():
                logger.warning("min_wage_store: invalid country code key: %r", k)
                continue
            try:
                parsed[cc] = MinWageRecord.model_validate(v)
            except Exception as e:  # noqa: BLE001
                logger.warning("min_wage_store: invalid record cc=%s err=%s", cc, e)
                continue

        _CACHED = parsed
        return _CACHED
    except FileNotFoundError:
        logger.warning("min_wage_store: dataset file not found: %s", path)
    except Exception as e:  # noqa: BLE001
        logger.warning("min_wage_store: failed to load dataset path=%s err=%s", path, e)

    _CACHED = {}
    return _CACHED


def get_min_wage_local(countryCode: str) -> tuple[float, str, str] | None:  # noqa: N802
    """
    Return (hourly, currency, asOf) for given ISO alpha-2 country code, else None.
    """
    cc = (countryCode or "").strip().upper()
    if len(cc) != 2 or not cc.isalpha():
        return None

    ds = load_min_wage_dataset()
    rec = ds.get(cc)
    if rec is None:
        return None
    return float(rec.hourly), str(rec.currency), str(rec.asOf)

