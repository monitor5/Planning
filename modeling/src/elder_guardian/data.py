"""Traceable loaders for population grids and the curated facility register."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import shapefile
from pyproj import Transformer


@dataclass(frozen=True)
class DataAudit:
    raw_rows: int
    unique_grids: int
    duplicate_rows: int
    published_positive_rows: int
    suppressed_rows: int
    blank_rows: int
    published_population: float


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_zipped_shapefile(path: Path) -> list[dict]:
    with ZipFile(path) as archive:
        shp_names = [name for name in archive.namelist() if name.lower().endswith(".shp")]
        if len(shp_names) != 1:
            raise ValueError(f"expected one shapefile in {path}, found {len(shp_names)}")
        base = shp_names[0][:-4]
        reader = shapefile.Reader(
            shp=BytesIO(archive.read(base + ".shp")),
            shx=BytesIO(archive.read(base + ".shx")),
            dbf=BytesIO(archive.read(base + ".dbf")),
            encoding="utf-8",
        )
        rows: list[dict] = []
        for item in reader.iterShapeRecords():
            record = item.record.as_dict()
            bbox = item.shape.bbox
            label = str(record.get("lbl") or "").strip()
            value = record.get("val")
            rows.append(
                {
                    "gid": str(record["gid"]),
                    "label": label,
                    "value": np.nan if value is None else float(value),
                    "x": (float(bbox[0]) + float(bbox[2])) / 2.0,
                    "y": (float(bbox[1]) + float(bbox[3])) / 2.0,
                }
            )
    return rows


def load_population_100m(root: str | Path) -> tuple[pd.DataFrame, DataAudit]:
    """Load all 25 district archives and sum cross-boundary gid contributions.

    A duplicated physical grid is not dropped: published district contributions
    are summed.  `N/A` suppression and blank DBF values remain explicit flags;
    the `population_lower_bound` column is therefore a published lower bound.
    """

    root_path = Path(root)
    archives = sorted(
        path
        for path in root_path.rglob("*.zip")
        if "100M" in path.name.upper() and "서울전체" not in str(path)
    )
    if len(archives) != 25:
        raise ValueError(f"expected 25 district 100m archives, found {len(archives)}")
    frames: list[pd.DataFrame] = []
    for archive in archives:
        frame = pd.DataFrame(_read_zipped_shapefile(archive))
        frame["borough"] = archive.parent.name
        frame["is_suppressed"] = frame["label"].str.upper().eq("N/A")
        frame["is_blank"] = frame["value"].isna() & ~frame["is_suppressed"]
        frame["published_value"] = frame["value"].fillna(0.0).clip(lower=0.0)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)

    dominant = (
        raw.sort_values(
            ["gid", "published_value", "borough"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("gid")
        .set_index("gid")["borough"]
    )
    grouped = raw.groupby("gid", sort=True, observed=True)
    population = grouped.agg(
        x=("x", "median"),
        y=("y", "median"),
        population_lower_bound=("published_value", "sum"),
        source_row_count=("gid", "size"),
        suppressed_source_rows=("is_suppressed", "sum"),
        blank_source_rows=("is_blank", "sum"),
    ).reset_index()
    population["borough"] = population["gid"].map(dominant)
    population["is_suppressed"] = population["suppressed_source_rows"] > 0
    population["has_blank_source"] = population["blank_source_rows"] > 0
    population["parent_1km_x"] = np.floor(population["x"] / 1000.0) * 1000.0
    population["parent_1km_y"] = np.floor(population["y"] / 1000.0) * 1000.0
    population["population"] = population["population_lower_bound"]

    audit = DataAudit(
        raw_rows=len(raw),
        unique_grids=len(population),
        duplicate_rows=len(raw) - len(population),
        published_positive_rows=int((raw["published_value"] > 0).sum()),
        suppressed_rows=int(raw["is_suppressed"].sum()),
        blank_rows=int(raw["is_blank"].sum()),
        published_population=float(population["population_lower_bound"].sum()),
    )
    return population, audit


def apply_1km_residual_sensitivity(
    population: pd.DataFrame, one_km_archive: str | Path
) -> tuple[pd.DataFrame, dict]:
    """Allocate only positive 1km residuals equally to explicitly suppressed cells.

    Blank cells are deliberately not imputed.  The result is a sensitivity
    scenario, not a reconstruction of confidential cell counts.
    """

    result = population.copy()
    coarse = pd.DataFrame(_read_zipped_shapefile(Path(one_km_archive)))
    coarse["parent_1km_x"] = np.floor(coarse["x"] / 1000.0) * 1000.0
    coarse["parent_1km_y"] = np.floor(coarse["y"] / 1000.0) * 1000.0
    coarse["coarse_value"] = coarse["value"].fillna(0.0).clip(lower=0.0)
    coarse_map = coarse.set_index(["parent_1km_x", "parent_1km_y"])["coarse_value"]
    lower_sum = result.groupby(["parent_1km_x", "parent_1km_y"])[
        "population_lower_bound"
    ].transform("sum")
    coarse_values = pd.MultiIndex.from_frame(
        result[["parent_1km_x", "parent_1km_y"]]
    ).map(coarse_map)
    result["coarse_1km_population"] = np.asarray(coarse_values, dtype=float)
    residual = np.maximum(result["coarse_1km_population"].fillna(0.0) - lower_sum, 0.0)
    suppressed_count = result.groupby(["parent_1km_x", "parent_1km_y"])[
        "is_suppressed"
    ].transform("sum")
    allocation = np.divide(
        residual,
        suppressed_count,
        out=np.zeros(len(result), dtype=float),
        where=suppressed_count.to_numpy() > 0,
    )
    result["population"] = result["population_lower_bound"] + np.where(
        result["is_suppressed"], allocation, 0.0
    )
    metadata = {
        "method": "positive_1km_residual_equal_to_explicit_NA_cells",
        "lower_bound_total": float(result["population_lower_bound"].sum()),
        "sensitivity_total": float(result["population"].sum()),
        "allocated_total": float(
            (result["population"] - result["population_lower_bound"]).sum()
        ),
    }
    return result, metadata


def load_facilities(
    master_csv: str | Path,
    *,
    historical_csv: str | Path | None = None,
    capacity_mode: str = "LEGAL_NOMINAL",
) -> tuple[pd.DataFrame, dict]:
    """Load P0 facilities with explicit non-operational capacity semantics."""

    master_path = Path(master_csv)
    frame = pd.read_csv(master_path, low_memory=False)
    eligible = frame["p0_service_eligible"].astype(str).str.lower().eq("true")
    frame = frame.loc[eligible].copy()
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame = frame.loc[
        frame["latitude"].between(37.3, 37.8)
        & frame["longitude"].between(126.7, 127.3)
    ].copy()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    x, y = transformer.transform(
        frame["longitude"].to_numpy(), frame["latitude"].to_numpy()
    )
    frame["x"] = x
    frame["y"] = y

    if capacity_mode == "LEGAL_NOMINAL":
        frame["model_capacity"] = pd.to_numeric(
            frame["capacity_legal_nominal"], errors="coerce"
        ).fillna(0.0)
        semantics = "legal minimum proxy; status unverified; sensitivity only"
    elif capacity_mode == "STRICT_UNKNOWN":
        frame["model_capacity"] = 0.0
        semantics = "verified operational capacity unavailable; baseline Gini undefined"
    elif capacity_mode == "HISTORICAL_REFERENCE_WINSORIZED":
        if historical_csv is None:
            raise ValueError("historical_csv is required for historical mode")
        history = pd.read_csv(historical_csv, low_memory=False)
        history["historical_capacity"] = pd.to_numeric(
            history["capacity_registered_historical_candidate"], errors="coerce"
        )
        history = history.loc[history["historical_capacity"] > 0]
        positive = history["historical_capacity"]
        upper = float(positive.quantile(0.99))
        history["historical_capacity"] = positive.clip(upper=upper)
        history = history.drop_duplicates("source_record_id")
        frame = frame.merge(
            history[["source_record_id", "historical_capacity"]],
            on="source_record_id",
            how="left",
        )
        nominal = pd.to_numeric(
            frame["capacity_legal_nominal"], errors="coerce"
        ).fillna(0.0)
        frame["model_capacity"] = frame["historical_capacity"].fillna(nominal)
        semantics = (
            "stale historical registered-admission reference winsorized at p99; "
            "not current operational capacity"
        )
    else:
        raise ValueError(f"unknown capacity mode: {capacity_mode}")

    metadata = {
        "capacity_mode": capacity_mode,
        "capacity_semantics": semantics,
        "facility_rows": int(len(frame)),
        "capacity_sum": float(frame["model_capacity"].sum()),
        "verified_operational_capacity_rows": int(
            pd.to_numeric(frame["capacity_operational"], errors="coerce").notna().sum()
        ),
        "source_sha256": sha256_file(master_path),
    }
    return frame.reset_index(drop=True), metadata

