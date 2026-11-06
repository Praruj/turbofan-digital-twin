"""NASA C-MAPSS turbofan run-to-failure data: download, checksum, load."""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

#: NASA Prognostics Center of Excellence data repository (public, U.S. Government work).
CMAPSS_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
CMAPSS_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"

#: Column names. Sensor names follow Saxena et al. (2008), Table 2.
SENSOR_NAMES: tuple[str, ...] = (
    "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30", "phi",
    "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32",
)  # fmt: skip
COLUMNS: tuple[str, ...] = ("unit", "cycle", "alt", "mach", "tra", *SENSOR_NAMES)

#: Sensors that carry degradation information in FD001/FD003 (the rest are constant).
INFORMATIVE: tuple[str, ...] = (
    "T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30", "phi", "NRf", "NRc", "BPR", "htBleed", "W31", "W32",
)  # fmt: skip


def download(raw_dir: str | Path) -> Path:
    """Fetch and unpack the C-MAPSS archive into ``raw_dir`` (idempotent, checksummed)."""
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    if (raw / "train_FD001.txt").exists():
        return raw
    with urllib.request.urlopen(CMAPSS_URL, timeout=120) as resp:
        blob = resp.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != CMAPSS_SHA256:
        raise RuntimeError(f"C-MAPSS checksum mismatch: {digest}")
    with zipfile.ZipFile(io.BytesIO(blob)) as outer:
        inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
            for name in inner.namelist():
                if name.endswith(".txt"):
                    (raw / Path(name).name).write_bytes(inner.read(name))
    return raw


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=list(COLUMNS))
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_train(raw_dir: str | Path, subset: str = "FD001") -> pd.DataFrame:
    """Training trajectories (run to failure) with a ``rul`` column."""
    df = _read(Path(raw_dir) / f"train_{subset}.txt")
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df


def load_test(raw_dir: str | Path, subset: str = "FD001") -> pd.DataFrame:
    """Test trajectories (truncated before failure) with the official ``rul`` per row."""
    raw = Path(raw_dir)
    df = _read(raw / f"test_{subset}.txt")
    truth = np.loadtxt(raw / f"RUL_{subset}.txt", dtype=int)
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = truth[df["unit"].to_numpy() - 1] + (last - df["cycle"])
    return df
