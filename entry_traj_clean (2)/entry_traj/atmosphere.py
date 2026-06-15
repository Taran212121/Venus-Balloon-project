"""Venus atmosphere: density/temperature models, VIRA profiles, GRAM dispersions.

Three layers:

- ``Profile`` / ``load_profile``: VCD-VIRA and Venus-GRAM VIRA text tables.
- ``GramSource`` / ``AtmosphereSample``: per-run density and temperature
  perturbations, either synthetic log-normal or rows from a Venus-GRAM HDF5.
- ``AtmosphereModel``: density, temperature and speed of sound at altitude for
  one run (base model plus an optional dispersion sample).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

from .models import Config
from .physics import interp, log_interp

VIRA_LAT_SELECTORS: tuple[str, ...] = ("lat0", "lat30", "lat45", "lat60", "lat75", "lat85")
VIRA_SZA_SELECTORS: tuple[str, ...] = ("sza15", "sza34", "sza53", "sza71", "sza90", "sza108", "sza165")


# --------------------------------------------------------------------------
# Profile tables
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    altitude_m: tuple[float, ...]
    density_kg_m3: tuple[float, ...]
    temperature_k: tuple[float, ...]
    source_id: str


def vira_selectors() -> tuple[str, ...]:
    """Full VCD-VIRA latitude x solar-zenith-angle selector grid."""
    return tuple(f"{lat}_{sza}" for lat in VIRA_LAT_SELECTORS for sza in VIRA_SZA_SELECTORS)


def load_profile(path: str | Path, profile_format: str, selector: str) -> Profile:
    return _load_profile_cached(str(Path(path)), profile_format, selector)


@lru_cache(maxsize=32)
def _load_profile_cached(path_text: str, profile_format: str, selector: str) -> Profile:
    path = Path(path_text)
    if profile_format == "vcd_vira":
        return _load_vcd_vira(path, selector)
    if profile_format == "venusgram_vira":
        return _load_venusgram_vira(path, selector)
    raise ValueError(f"Unsupported atmosphere profile_format {profile_format!r}")


def _load_vcd_vira(path: Path, selector: str) -> Profile:
    if path.is_file():
        files: tuple[Path, ...] = (path,)
    else:
        files = tuple(
            item
            for item in (
                path / "VIRA_z_0_100.txt",
                path / "VIRA_z_100_150.txt",
                path / "VIRA_z_150_250.txt",
            )
            if item.exists()
        )
    rows: dict[float, tuple[float, float]] = {}
    for file_path in files:
        name = file_path.name.lower()
        if "0_100" in name:
            rows.update(_read_vira_0_100(file_path, selector))
        elif "100_150" in name:
            rows.update(_read_vira_100_150(file_path, selector))
        elif "150_250" in name:
            rows.update(_read_vira_150_250(file_path, selector))
    if len(rows) < 2:
        raise ValueError(f"No usable VIRA rows found in {path}")
    ordered = sorted(rows.items())
    return Profile(
        altitude_m=tuple(alt for alt, _ in ordered),
        density_kg_m3=tuple(values[1] for _, values in ordered),
        temperature_k=tuple(values[0] for _, values in ordered),
        source_id=f"VCD-VIRA:{selector}",
    )


def _read_vira_0_100(path: Path, selector: str) -> dict[float, tuple[float, float]]:
    idx = VIRA_LAT_SELECTORS.index(_selector_lat(selector))
    rows: dict[float, tuple[float, float]] = {}
    for values in _numeric_rows(path):
        if len(values) >= 19:
            rows[values[0]] = (values[1 + idx], values[13 + idx])
    return rows


def _read_vira_100_150(path: Path, selector: str) -> dict[float, tuple[float, float]]:
    night = "night" in selector.lower()
    rows: dict[float, tuple[float, float]] = {}
    for values in _numeric_rows(path):
        if len(values) >= 7:
            rows[values[0]] = (values[2], values[6]) if night else (values[1], values[5])
    return rows


def _read_vira_150_250(path: Path, selector: str) -> dict[float, tuple[float, float]]:
    idx = VIRA_SZA_SELECTORS.index(_selector_sza(selector))
    rows: dict[float, tuple[float, float]] = {}
    for values in _numeric_rows(path):
        if len(values) >= 22:
            rows[values[0]] = (values[1 + idx], values[15 + idx])
    return rows


def _load_venusgram_vira(path: Path, selector: str) -> Profile:
    file_path = path if path.is_file() else path / "VIRAMid.txt"
    target_lst = 12.0 if any(tok in selector.lower() for tok in ("day", "lst12")) else 0.0
    rows: dict[float, tuple[float, float]] = {}
    for values in _numeric_rows(file_path):
        if len(values) < 12 or abs(values[1] - target_lst) > 0.1:
            continue
        rows[values[0] * 1000.0] = (values[10], values[8])
    if len(rows) < 2:
        raise ValueError(f"No usable Venus-GRAM VIRA rows found in {file_path}")
    ordered = sorted(rows.items())
    return Profile(
        altitude_m=tuple(alt for alt, _ in ordered),
        density_kg_m3=tuple(values[1] for _, values in ordered),
        temperature_k=tuple(values[0] for _, values in ordered),
        source_id=f"Venus-GRAM-VIRA:{selector}",
    )


def _numeric_rows(path: Path) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                rows.append(tuple(float(part.replace("D", "E")) for part in stripped.split()))
            except ValueError:
                continue
    return rows


def _selector_lat(selector: str) -> str:
    lowered = selector.lower()
    for lat in VIRA_LAT_SELECTORS:
        if lat in lowered:
            return lat
    return "lat75" if "north" in lowered else "lat0"


def _selector_sza(selector: str) -> str:
    lowered = selector.lower()
    for sza in VIRA_SZA_SELECTORS:
        if sza in lowered:
            return sza
    return "sza165" if "night" in lowered else "sza15"


# --------------------------------------------------------------------------
# Dispersion samples
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AtmosphereSample:
    """One dispersed atmosphere realization.

    Scalar multipliers apply uniformly; when altitude-resolved arrays are
    present (Venus-GRAM HDF5 mode) they take precedence.
    """

    rho_sigma: float = 0.0
    density_multiplier: float = 1.0
    temperature_delta_k: float = 0.0
    gram_sample_index: int | None = None
    profile_altitude_m: tuple[float, ...] = ()
    profile_density_multiplier: tuple[float, ...] = ()
    profile_temperature_delta_k: tuple[float, ...] = ()

    def density_multiplier_at(self, altitude_m: float) -> float:
        if self.profile_altitude_m and self.profile_density_multiplier:
            return interp(altitude_m, self.profile_altitude_m, self.profile_density_multiplier)
        return self.density_multiplier

    def temperature_delta_at(self, altitude_m: float) -> float:
        if self.profile_altitude_m and self.profile_temperature_delta_k:
            return interp(altitude_m, self.profile_altitude_m, self.profile_temperature_delta_k)
        return self.temperature_delta_k


class GramSource:
    """Maps a uniform unit sample to an ``AtmosphereSample``.

    ``synthetic_gram_fixture``: log-normal density multiplier and normal
    temperature delta (Phase-0 placeholder, G5 residual).
    ``venus_gram_hdf5``: rows from a Venus-GRAM dispersion file via h5py.
    """

    def __init__(self, cfg: Config) -> None:
        self._dispersion = cfg.atmosphere.dispersion
        self._gram_rows: list[AtmosphereSample] | None = None
        if self._dispersion.mode == "venus_gram_hdf5":
            self._gram_rows = _load_gram_rows(self._dispersion)

    @classmethod
    def from_config(cls, cfg: Config) -> "GramSource":
        return cls(cfg)

    def sample_from_unit(self, unit: float) -> AtmosphereSample:
        u = min(1.0 - 1e-12, max(1e-12, unit))
        if self._gram_rows is not None:
            index = min(len(self._gram_rows) - 1, int(u * len(self._gram_rows)))
            return self._gram_rows[index]
        z = NormalDist().inv_cdf(u)
        return AtmosphereSample(
            rho_sigma=z,
            density_multiplier=math.exp(self._dispersion.synthetic_density_ln_sigma * z),
            temperature_delta_k=self._dispersion.synthetic_temperature_sigma_k * z,
        )


def _load_gram_rows(dispersion) -> list[AtmosphereSample]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for venus_gram_hdf5 dispersion mode.") from exc
    if dispersion.venus_gram_hdf5_path is None:
        raise ValueError("venus_gram_hdf5 mode requires venus_gram_hdf5_path")

    rows: list[AtmosphereSample] = []
    with h5py.File(dispersion.venus_gram_hdf5_path, "r") as handle:
        multipliers = handle[dispersion.density_multiplier_dataset][()]
        altitudes = (
            tuple(float(v) for v in handle[dispersion.altitude_dataset][()])
            if dispersion.altitude_dataset and dispersion.altitude_dataset in handle
            else ()
        )
        deltas = (
            handle[dispersion.temperature_delta_dataset][()]
            if dispersion.temperature_delta_dataset
            and dispersion.temperature_delta_dataset in handle
            else None
        )
        ln_sigma = max(1e-9, dispersion.synthetic_density_ln_sigma)
        for index in range(multipliers.shape[0]):
            row = multipliers[index]
            if getattr(row, "ndim", 0) == 0:  # scalar multiplier per sample
                multiplier = float(row)
                delta = float(deltas[index]) if deltas is not None else 0.0
                rows.append(
                    AtmosphereSample(
                        rho_sigma=math.log(max(multiplier, 1e-12)) / ln_sigma,
                        density_multiplier=multiplier,
                        temperature_delta_k=delta,
                        gram_sample_index=index,
                    )
                )
            else:  # altitude-resolved multiplier profile
                profile = tuple(float(v) for v in row)
                mean_multiplier = sum(profile) / len(profile)
                delta_profile = (
                    tuple(float(v) for v in deltas[index]) if deltas is not None else ()
                )
                rows.append(
                    AtmosphereSample(
                        rho_sigma=math.log(max(mean_multiplier, 1e-12)) / ln_sigma,
                        density_multiplier=mean_multiplier,
                        temperature_delta_k=(
                            sum(delta_profile) / len(delta_profile) if delta_profile else 0.0
                        ),
                        gram_sample_index=index,
                        profile_altitude_m=altitudes,
                        profile_density_multiplier=profile,
                        profile_temperature_delta_k=delta_profile,
                    )
                )
    if not rows:
        raise ValueError("Venus-GRAM HDF5 file contained no dispersion rows")
    return rows


# --------------------------------------------------------------------------
# Atmosphere model
# --------------------------------------------------------------------------


class AtmosphereModel:
    """Density, temperature and speed of sound at altitude for one run."""

    def __init__(self, cfg: Config, sample: AtmosphereSample | None = None) -> None:
        self._cfg = cfg
        self._atm = cfg.atmosphere
        self._sample = sample
        self._profile: Profile | None = None
        if self._atm.density_kind == "profile_file":
            if (
                self._atm.profile_path is None
                or self._atm.profile_format is None
                or self._atm.profile_selector is None
            ):
                raise ValueError("profile_file density requires path, format and selector")
            self._profile = load_profile(
                self._atm.profile_path, self._atm.profile_format, self._atm.profile_selector
            )

    def density_kg_m3(self, altitude_m: float) -> float:
        atm = self._atm
        if self._profile is not None:
            base = log_interp(altitude_m, self._profile.altitude_m, self._profile.density_kg_m3)
        elif atm.density_kind == "table":
            base = log_interp(altitude_m, atm.density_altitude_m, atm.density_kg_m3)
        else:
            base = atm.rho_ref_kg_m3 * math.exp(-(altitude_m - atm.alt_ref_m) / atm.scale_height_m)
        if self._sample is not None:
            base *= self._sample.density_multiplier_at(altitude_m)
        return max(0.0, base)

    def temperature_k(self, altitude_m: float) -> float:
        temp = self._atm.temperature
        if temp.kind == "constant" and temp.constant_temperature_k is not None:
            base = temp.constant_temperature_k
        elif temp.kind == "table" and temp.altitude_m:
            base = interp(altitude_m, temp.altitude_m, temp.temperature_k)
        elif self._profile is not None:
            base = interp(altitude_m, self._profile.altitude_m, self._profile.temperature_k)
        else:
            base = 200.0
        if self._sample is not None:
            base += self._sample.temperature_delta_at(altitude_m)
        return max(1.0, base)

    def speed_of_sound_m_s(self, altitude_m: float) -> float:
        temp = self._atm.temperature
        if temp.kind == "fallback_speed_of_sound" and temp.fallback_speed_of_sound_m_s is not None:
            return temp.fallback_speed_of_sound_m_s
        return math.sqrt(self._cfg.gas_gamma * self._cfg.gas_constant_j_kg_k * self.temperature_k(altitude_m))
