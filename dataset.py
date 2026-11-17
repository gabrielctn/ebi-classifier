"""Dataset utilities for ImageStream/IFC OME-TIFF triplets.

This module implements:
- Indexing of per-object TIFF triplets (one file per channel) into `TripletSample`.
- A PyTorch `Dataset` that loads the 3 channels and maps them into an RGB-like tensor.
- Preprocessing (letterbox resize+pad) and augmentation policies tuned for microscopy.

Assumptions (based on the dataset in this workspace):
- Each object/sample is exported as multiple TIFF files sharing the same sample id.
- Channel id is encoded in the filename like `<id>_Ch11.ome.tif` (case-insensitive).
- The model is trained on exactly 3 channels, mapped to RGB in a fixed order.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import tifffile
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "tifffile is required to read OME-TIFFs. Install it with: uv pip install tifffile"
    ) from e

try:
    from torchvision.transforms import v2
    from torchvision.transforms.functional import InterpolationMode
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torchvision is required. Install it with: uv pip install torchvision"
    ) from e


REQUIRED_CHANNELS_DEFAULT: Tuple[int, int, int] = (11, 7, 3)

# Default pseudo-RGB mapping used in this workspace:
#   R <- Ch7   (erythroblasts)
#   G <- Ch11  (macrophages)
#   B <- Ch3   (neutrophils)
#
# Note: Some writeups swap R/G. To avoid ambiguity, the Dataset also supports
# overriding the mapping via the `rgb_channel_order` argument.
RGB_CHANNEL_ORDER: Tuple[int, int, int] = (7, 11, 3)


# The ImageStream OME-TIFF exports in this workspace often contain metadata that triggers
# very noisy tifffile warnings (e.g., OME series discontiguous storage). We do not rely
# on OME parsing here, so we silence the tifffile logger and read the first page directly.
logging.getLogger("tifffile").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class RandomGamma(nn.Module):
    """Apply a random gamma curve to an image tensor in [0,1].

    This is a lightweight intensity augmentation that is usually safe for microscopy
    (it preserves structure while altering contrast).
    """

    def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5) -> None:
        super().__init__()
        self.gamma_range = gamma_range
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0:
            return x
        if torch.rand((), device=x.device) > self.p:
            return x
        g0, g1 = self.gamma_range
        gamma = g0 + (g1 - g0) * torch.rand((), device=x.device)
        return x.clamp(0.0, 1.0).pow(gamma)


class RandomGaussianNoise(nn.Module):
    """Add random i.i.d. Gaussian noise to an image tensor in [0,1]."""

    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.03), p: float = 0.5) -> None:
        super().__init__()
        self.std_range = std_range
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0:
            return x
        if torch.rand((), device=x.device) > self.p:
            return x
        s0, s1 = self.std_range
        std = s0 + (s1 - s0) * torch.rand((), device=x.device)
        if float(std) <= 0.0:
            return x
        noise = torch.randn_like(x) * std
        return (x + noise).clamp(0.0, 1.0)


class RandomRotate90(nn.Module):
    """Rotate an image tensor by k*90 degrees.

    This avoids interpolation artifacts compared to arbitrary-angle rotation.
    Expects CHW tensors.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0:
            return x
        if torch.rand((), device=x.device) > self.p:
            return x
        k = int(torch.randint(low=0, high=4, size=(), device=x.device).item())
        if k == 0:
            return x
        return torch.rot90(x, k=k, dims=(-2, -1))


class RandomCoarseDropout(nn.Module):
    """Randomly zero out small square patches (Cutout/CoarseDropout-like)."""

    def __init__(
        self,
        max_holes: int = 8,
        max_size: int = 8,
        p: float = 0.25,
    ) -> None:
        super().__init__()
        self.max_holes = int(max_holes)
        self.max_size = int(max_size)
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or self.max_holes <= 0 or self.max_size <= 0:
            return x
        if torch.rand((), device=x.device) > self.p:
            return x

        if x.ndim != 3:
            raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")
        _, h, w = x.shape
        if h <= 1 or w <= 1:
            return x

        holes = int(torch.randint(low=1, high=self.max_holes + 1, size=()).item())
        for _ in range(holes):
            size = int(torch.randint(low=1, high=self.max_size + 1, size=()).item())
            cy = int(torch.randint(low=0, high=h, size=()).item())
            cx = int(torch.randint(low=0, high=w, size=()).item())
            y0 = max(0, cy - size // 2)
            y1 = min(h, y0 + size)
            x0 = max(0, cx - size // 2)
            x1 = min(w, x0 + size)
            x[:, y0:y1, x0:x1] = 0.0
        return x


def _maybe_get_torchvision_aug(name: str):
    """Return an augmentation transform from torchvision if available.

    Torchvision's v2 API has changed over time; this helper keeps optional policies
    (e.g., RandAugment) best-effort.
    """

    return getattr(v2, name, None)


@dataclass(frozen=True)
class TripletSample:
    """A single object represented by multiple single-channel TIFF files."""

    sample_id: str
    label: int  # 1 = EBI, 0 = NOT_EBI
    channels: Dict[int, Path]  # channel_id -> file path


_CHANNEL_RE = re.compile(r"(?:^|[_\-])ch(?P<ch>\d+)(?:\D|$)", re.IGNORECASE)


def _is_tiff(path: Path) -> bool:
    """Return True if a filename extension looks like a TIFF/OME-TIFF."""
    name = path.name.lower()
    return name.endswith(".tif") or name.endswith(".tiff") or name.endswith(".ome.tif") or name.endswith(
        ".ome.tiff"
    )


def _looks_like_tiff(path: Path) -> bool:
    """Cheap validation that a file is (likely) a TIFF.

    This catches empty/corrupted files that have the right extension but no TIFF header.
    """

    try:
        if path.stat().st_size < 4:
            return False
        with path.open("rb") as f:
            header = f.read(4)
    except OSError:
        return False

    if len(header) < 4:
        return False

    # Classic TIFF / BigTIFF signatures
    if header[:2] == b"II" and header[2:4] in {b"*\x00", b"+\x00"}:
        return True
    if header[:2] == b"MM" and header[2:4] in {b"\x00*", b"\x00+"}:
        return True
    # tifffile also allows b"EP" for some variants
    if header[:2] == b"EP":
        return True

    return False


def _parse_channel_id(filename: str) -> Optional[int]:
    """Extract the channel number from a filename (e.g. `_Ch11` -> 11)."""
    m = _CHANNEL_RE.search(filename)
    if not m:
        return None
    try:
        return int(m.group("ch"))
    except ValueError:
        return None


def _parse_sample_id(filename: str) -> str:
    """Extract the sample id for grouping per-object TIFFs.

    Primary expected form is `<id>_Ch{n}...tif`, where `<id>` is returned.
    A fallback strips the channel token and extension.
    """

    # Primary: match ImageStream export like: 10050_Ch11.ome.tif -> "10050"
    lower = filename.lower()
    idx = lower.find("_ch")
    if idx > 0:
        return filename[:idx]
    # Fallback: strip channel token (best-effort)
    cleaned = _CHANNEL_RE.sub("_", filename)
    for suffix in (".ome.tif", ".ome.tiff", ".tif", ".tiff"):
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned.strip("_-")


def build_triplet_index(
    ebi_dir: Path | str,
    not_ebi_dir: Path | str,
    required_channels: Sequence[int] = REQUIRED_CHANNELS_DEFAULT,
    strict: bool = True,
    validate_files: bool = True,
) -> List[TripletSample]:
    """Scan the two class folders and build a list of samples.

    Matching logic:
      - Groups files by sample id (basename before "_Ch")
      - Extracts the channel id from "_Ch{n}" (case-insensitive)
      - Keeps only samples that contain all required channels when strict=True

    Expected export naming (observed in this workspace):
      <id>_Ch11.ome.tif, <id>_Ch7.ome.tif, <id>_Ch3.ome.tif
    """

    ebi_dir = Path(ebi_dir)
    not_ebi_dir = Path(not_ebi_dir)

    required_set = set(int(c) for c in required_channels)

    def scan_class_dir(class_dir: Path, label: int) -> Dict[str, Dict[int, Path]]:
        groups: Dict[str, Dict[int, Path]] = {}
        for p in class_dir.iterdir():
            if not p.is_file() or not _is_tiff(p):
                continue
            if validate_files and not _looks_like_tiff(p):
                logger.warning("Skipping non-TIFF/corrupted file: %s", p)
                continue
            ch = _parse_channel_id(p.name)
            if ch is None:
                continue
            if ch not in required_set:
                continue
            sid = _parse_sample_id(p.name)
            groups.setdefault(sid, {})[ch] = p
        return groups

    ebi_groups = scan_class_dir(ebi_dir, label=1)
    not_groups = scan_class_dir(not_ebi_dir, label=0)

    samples: List[TripletSample] = []

    skipped_incomplete = 0
    def add_groups(groups: Dict[str, Dict[int, Path]], label: int) -> None:
        for sid, ch_map in groups.items():
            nonlocal skipped_incomplete
            if strict and (set(ch_map.keys()) != required_set):
                skipped_incomplete += 1
                continue
            # Even in non-strict mode, we still require at least one channel.
            if not ch_map:
                continue
            samples.append(TripletSample(sample_id=sid, label=label, channels=ch_map))

    add_groups(ebi_groups, label=1)
    add_groups(not_groups, label=0)

    if skipped_incomplete > 0:
        logger.info("Skipped %d incomplete triplets (missing required channels)", skipped_incomplete)

    # Deterministic ordering
    samples.sort(key=lambda s: (s.label, s.sample_id))
    return samples


def _read_ome_tiff_2d(path: Path) -> np.ndarray:
    """Read a single-channel TIFF as a 2D array.

    We intentionally bypass OME series parsing because these exports can contain
    discontiguous storage that OME-series readers complain about.
    """

    if not _looks_like_tiff(path):
        raise tifffile.TiffFileError(f"File does not look like a TIFF (empty/corrupt?): {path}")

    try:
        with tifffile.TiffFile(str(path)) as tif:
            arr = tif.pages[0].asarray()
    except Exception:
        # Fallback: explicitly disable OME handling
        arr = tifffile.imread(str(path), is_ome=False)

    arr = np.asarray(arr)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported TIFF shape {arr.shape} for {path}")
    return arr


def _to_float01(
    arr: np.ndarray,
    *,
    intensity_norm: str = "dtype",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> torch.Tensor:
    """Convert a numpy image array into float32 torch tensor in [0,1].

    intensity_norm:
      - "dtype": scale integers by dtype max (fast, stable)
      - "percentile": clip to [p_low, p_high] then rescale (better for 16-bit dynamic range)
    """

    intensity_norm = str(intensity_norm).lower().strip()

    arr_np = np.asarray(arr)
    if intensity_norm == "percentile":
        x = arr_np.astype(np.float32, copy=False)
        x = np.maximum(x, 0.0)
        lo = float(np.percentile(x, percentile_low))
        hi = float(np.percentile(x, percentile_high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # Fallback to max scaling
            m = float(np.max(x))
            if m <= 0.0 or not np.isfinite(m):
                return torch.from_numpy(np.zeros_like(x, dtype=np.float32))
            x = x / max(1e-6, m)
        else:
            x = (x - lo) / max(1e-6, (hi - lo))
        return torch.from_numpy(np.clip(x, 0.0, 1.0).astype(np.float32, copy=False))

    # Default: dtype scaling
    if np.issubdtype(arr_np.dtype, np.floating):
        x_t = torch.from_numpy(arr_np.astype(np.float32, copy=False)).clamp_min(0.0)
        if float(x_t.max()) > 1.0:
            x_t = x_t / (x_t.max().clamp_min(1e-6))
        return x_t

    if np.issubdtype(arr_np.dtype, np.integer):
        info = np.iinfo(arr_np.dtype)
        x_t = torch.from_numpy(arr_np.astype(np.float32, copy=False))
        return (x_t / float(info.max)).clamp(0.0, 1.0)

    x_t = torch.from_numpy(arr_np.astype(np.float32, copy=False)).clamp_min(0.0)
    if float(x_t.max()) > 1.0:
        x_t = x_t / (x_t.max().clamp_min(1e-6))
    return x_t


def pad_to_square(x: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """Pad CHW tensor to a square (max(H,W)) using constant fill."""
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")

    c, h, w = x.shape
    if h == w:
        return x

    if h > w:
        diff = h - w
        pad_left = diff // 2
        pad_right = diff - pad_left
        pad = (pad_left, pad_right, 0, 0)
    else:
        diff = w - h
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        pad = (0, 0, pad_top, pad_bottom)

    return F.pad(x, pad=pad, mode="constant", value=fill)


def resize_square(x: torch.Tensor, size: int) -> torch.Tensor:
    """Resize a CHW tensor to `size x size` using bilinear interpolation."""
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")
    # v2.Resize works on tensors, but we want explicit square + antialias.
    return v2.Resize(
        size=(size, size),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )(x)


def letterbox_resize_pad(x: torch.Tensor, size: int, fill: float = 0.0) -> torch.Tensor:
    """Resize while preserving aspect ratio, then pad to a square canvas.

    This implements the "letterbox" strategy for ImageStream variable-height inputs.
    Expects CHW and returns CHW of shape [C, size, size].
    """

    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")

    c, h, w = x.shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {tuple(x.shape)}")

    size = int(size)
    r = min(size / float(h), size / float(w))
    new_h = max(1, int(round(h * r)))
    new_w = max(1, int(round(w * r)))

    x_resized = v2.Resize(
        size=(new_h, new_w),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )(x)

    pad_h = size - new_h
    pad_w = size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return F.pad(x_resized, pad=(pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=float(fill))


def build_transforms(
    train: bool,
    image_size: int,
    normalize: str = "imagenet",
    augment: str = "basic",
) -> v2.Compose:
    """Build preprocessing + augmentation transforms.

    Pipeline shape invariants:
    - Input is CHW with 3 channels (RGB-mapped microscopy channels).
    - Output is CHW with shape [3, image_size, image_size].

    Normalization:
    - `imagenet`: standard ImageNet mean/std (works well with timm pretrained backbones)
    - `none`: keep in [0,1] without normalization
    """

    augment = augment.lower().strip()

    # Letterbox always: preserve aspect ratio (critical for ImageStream 120xN strips).
    tfs: List[torch.nn.Module] = [v2.Lambda(lambda x: letterbox_resize_pad(x, size=image_size, fill=0.0))]

    if train:
        # Classical microscopy-safe augmentations: geometry + mild intensity + noise.
        if augment in {"basic", "strong", "sota"}:
            tfs += [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                v2.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05) if augment == "basic" else (0.10, 0.10),
                    scale=(0.95, 1.05) if augment == "basic" else (0.90, 1.10),
                    shear=0.0,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                ),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15 if augment == "basic" else 0.25),
                v2.RandomApply([v2.ColorJitter(brightness=0.12, contrast=0.12)], p=0.5 if augment == "basic" else 0.7),
                RandomGamma(gamma_range=(0.85, 1.15) if augment == "basic" else (0.75, 1.30), p=0.5),
                RandomGaussianNoise(std_range=(0.0, 0.02) if augment == "basic" else (0.0, 0.04), p=0.4 if augment == "basic" else 0.6),
            ]

            if augment in {"strong", "sota"}:
                tfs += [
                    RandomCoarseDropout(max_holes=8, max_size=8, p=0.25),
                ]

        elif augment in {"none", "no"}:
            pass
        else:
            raise ValueError("augment must be one of: none, basic, strong, sota")

        # For SOTA-style policies, optionally stack a modern augmentation policy.
        if augment == "sota":
            # Prefer RandAugment if available, otherwise TrivialAugmentWide.
            RandAugment = _maybe_get_torchvision_aug("RandAugment")
            TrivialAugmentWide = _maybe_get_torchvision_aug("TrivialAugmentWide")
            if RandAugment is not None:
                tfs += [RandAugment(num_ops=2, magnitude=9)]
            elif TrivialAugmentWide is not None:
                tfs += [TrivialAugmentWide()]

    else:
        # Validation/test: deterministic (letterbox already applied)
        pass

    if normalize.lower() == "imagenet":
        tfs += [
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ]
    elif normalize.lower() in {"none", "no", "false"}:
        pass
    else:
        raise ValueError("normalize must be 'imagenet' or 'none'")

    return v2.Compose(tfs)


class ImagingFlowCytometryTripletDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for ImageStream exports where each object is 3 TIFFs.

        Channel mapping (pseudo-RGB, configurable via `rgb_channel_order`):
            - rgb_channel_order[0] -> R (channel 0)
            - rgb_channel_order[1] -> G (channel 1)
            - rgb_channel_order[2] -> B (channel 2)

    Returns:
      image: FloatTensor [3, image_size, image_size]
      label: FloatTensor [] (0.0 or 1.0)
    """

    def __init__(
        self,
        samples: Sequence[TripletSample],
        image_size: int = 224,
        required_channels: Sequence[int] = REQUIRED_CHANNELS_DEFAULT,
        rgb_channel_order: Sequence[int] = RGB_CHANNEL_ORDER,
        train: bool = True,
        normalize: str = "imagenet",
        augment: str = "basic",
        intensity_norm: str = "percentile",
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
    ) -> None:
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.required_channels = tuple(int(c) for c in required_channels)
        self.rgb_channel_order = tuple(int(c) for c in rgb_channel_order)
        self.intensity_norm = str(intensity_norm)
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)
        self.transforms = build_transforms(
            train=train,
            image_size=self.image_size,
            normalize=normalize,
            augment=augment,
        )

        missing = [
            s.sample_id
            for s in self.samples
            if any(ch not in s.channels for ch in self.required_channels)
        ]
        if missing:
            raise ValueError(
                f"Dataset contains {len(missing)} incomplete triplets (e.g. {missing[0]}). "
                "Rebuild the index with strict=True or clean the folder."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # Load channels and stack as pseudo-RGB (configurable channel order)
        missing_rgb = [ch for ch in self.rgb_channel_order if ch not in sample.channels]
        if missing_rgb:
            raise ValueError(
                f"Sample {sample.sample_id} is missing required channels for RGB mapping: {missing_rgb}"
            )

        rgb_tensors: List[torch.Tensor] = []
        for ch in self.rgb_channel_order:
            arr = _read_ome_tiff_2d(sample.channels[ch])
            rgb_tensors.append(
                _to_float01(
                    arr,
                    intensity_norm=self.intensity_norm,
                    percentile_low=self.percentile_low,
                    percentile_high=self.percentile_high,
                )
            )  # HW, float32

        img = torch.stack(rgb_tensors, dim=0)
        img = self.transforms(img)

        label = torch.tensor(float(sample.label), dtype=torch.float32)
        return img, label
