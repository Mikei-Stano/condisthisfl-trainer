# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ENVI format helpers
ENVI_DTYPE = {
    1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64,
    12: np.uint16, 13: np.uint32, 14: np.int64, 15: np.uint64,
}

# Class definitions
CLASSES = ["nizka_vegetacia", "stredna_vegetacia", "vysoka_vegetacia"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}


def parse_envi_hdr(hdr_path: Path) -> Dict:
    """Parse ENVI header file."""
    txt = hdr_path.read_text(errors="ignore")
    txt = re.sub(r"^\s*ENVI\s*", "", txt, flags=re.IGNORECASE)
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    lines, buf, in_brace = [], [], False
    for line in txt.split("\n"):
        if "{" in line and "}" not in line:
            in_brace = True
            buf.append(line)
            continue
        if in_brace:
            buf.append(line)
            if "}" in line:
                lines.append(" ".join(buf))
                buf = []
                in_brace = False
            continue
        lines.append(line)
    if buf:
        lines.append(" ".join(buf))
    d = {}
    for line in lines:
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().lower()
        val = val.strip()
        if val.startswith("{") and val.endswith("}"):
            val = val[1:-1].strip()
        d[key] = val
    for k in ["samples", "lines", "bands", "data type", "byte order"]:
        if k in d:
            try:
                d[k] = int(d[k])
            except:
                pass
    if "interleave" in d:
        d["interleave"] = d["interleave"].strip().lower()
    return d


def read_bsq_chip(bsq_path: Path) -> np.ndarray:
    """Read BSQ format hyperspectral chip."""
    hdr_path = bsq_path.with_suffix(".hdr")
    if not hdr_path.exists():
        alt = Path(str(bsq_path) + ".hdr")
        if alt.exists():
            hdr_path = alt
        else:
            raise FileNotFoundError(f"No .hdr found for {bsq_path}")
    
    meta = parse_envi_hdr(hdr_path)
    for k in ["samples", "lines", "bands", "data type", "byte order", "interleave"]:
        if k not in meta:
            raise ValueError(f"Missing '{k}' in {hdr_path}")
    
    if meta["interleave"] != "bsq":
        raise ValueError(f"Only BSQ supported; got {meta['interleave']} for {bsq_path}")
    
    samples, lines, bands = meta["samples"], meta["lines"], meta["bands"]
    dt_code = int(meta["data type"])
    if dt_code not in ENVI_DTYPE:
        raise ValueError(f"Unsupported ENVI data type {dt_code} in {hdr_path}")
    
    dtype = ENVI_DTYPE[dt_code]
    byte_order = "<" if int(meta["byte order"]) == 0 else ">"
    count = samples * lines * bands
    
    with open(bsq_path, "rb") as f:
        arr = np.fromfile(f, dtype=byte_order + np.dtype(dtype).str[1:], count=count)
    
    if arr.size != count:
        raise ValueError(f"Unexpected file size for {bsq_path}: expected {count}, got {arr.size}")
    
    arr = arr.reshape((bands, lines, samples)).astype(np.float32, copy=False)  # (C, H, W)
    return arr


def find_bsq_with_meta(root: Path) -> pd.DataFrame:
    """Find all BSQ chips with metadata."""
    rows = []
    for pdir in sorted(root.glob("P*")):
        if not pdir.is_dir():
            continue
        for section in sorted(pdir.iterdir()):
            if not section.is_dir():
                continue
            for cls in CLASSES:
                chips_dir = section / cls / "chips"
                if not chips_dir.exists():
                    continue
                for bsq in sorted(chips_dir.glob("*.bsq")):
                    hdr1 = bsq.with_suffix(".hdr")
                    hdr2 = Path(str(bsq) + ".hdr")
                    hdr = hdr1 if hdr1.exists() else (hdr2 if hdr2.exists() else None)
                    if hdr is None:
                        continue
                    try:
                        meta = parse_envi_hdr(hdr)
                        bands = int(meta.get("bands", 0))
                        lines = int(meta.get("lines", 0))
                        samples = int(meta.get("samples", 0))
                    except Exception:
                        continue
                    rows.append({
                        "parent": pdir.name,
                        "section": section.name,
                        "class": cls,
                        "bsq": str(bsq),
                        "bands": bands,
                        "lines": lines,
                        "samples": samples,
                    })
    return pd.DataFrame(rows)


def center_crop_or_pad(chip: np.ndarray, H: int, W: int) -> np.ndarray:
    """Center crop or pad chip to target size."""
    _, h, w = chip.shape
    # crop if larger
    if h > H:
        top = (h - H) // 2
        chip = chip[:, top:top+H, :]
        h = H
    if w > W:
        left = (w - W) // 2
        chip = chip[:, :, left:left+W]
        w = W
    # pad if smaller
    if h < H:
        pad_top = (H - h) // 2
        pad_bottom = H - h - pad_top
        chip = np.pad(chip, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=0.0)
        h = H
    if w < W:
        pad_left = (W - w) // 2
        pad_right = W - w - pad_left
        chip = np.pad(chip, ((0, 0), (0, 0), (pad_left, pad_right)), mode="constant", constant_values=0.0)
        w = W
    return chip


class BSQTreeDataset(Dataset):
    """
    Dataset for BSQ format hyperspectral tree chips.
    
    Args:
        table: DataFrame with columns ["bsq", "class", "bands", "lines", "samples"]
        band_norm: Dictionary with "mean" and "std" arrays for per-band normalization
        augment: Whether to apply data augmentation
        force_k: Force number of bands (channels)
        target_hw: Target height and width (H, W)
        local_class_id: The class ID that this client specializes in (for labeling)
    """
    def __init__(
        self, 
        table: pd.DataFrame, 
        band_norm=None, 
        augment: bool = False, 
        force_k: int = None, 
        target_hw: Tuple[int, int] = None,
        local_class_id: int = None,
    ):
        self.table = table.reset_index(drop=True)
        self.band_norm = band_norm
        self.augment = augment
        self.force_k = force_k  # K bands
        self.target_hw = target_hw  # (H, W)
        self.local_class_id = local_class_id  # Class this client specializes in

        if len(self.table) == 0:
            self.C = None
        else:
            first = read_bsq_chip(Path(self.table.loc[0, "bsq"]))  # (C0, h0, w0)
            C0 = first.shape[0]
            self.C = min(C0, self.force_k) if self.force_k is not None else C0
            
            # Estimate normalization (per-band) on up to 50 chips
            if band_norm is None:
                k = min(50, len(self.table))
                idxs = random.sample(range(len(self.table)), k)
                stack_mean = np.zeros(self.C, dtype=np.float64)
                stack_var = np.zeros(self.C, dtype=np.float64)
                n = 0
                for i in idxs:
                    chip = read_bsq_chip(Path(self.table.loc[i, "bsq"]))  # (C, h, w)
                    # channel fix
                    if self.force_k is not None:
                        if chip.shape[0] >= self.C:
                            chip = chip[:self.C, :, :]
                        else:
                            pad = np.zeros((self.C - chip.shape[0], chip.shape[1], chip.shape[2]), dtype=chip.dtype)
                            chip = np.concatenate([chip, pad], axis=0)
                    # spatial fix
                    if self.target_hw is not None:
                        chip = center_crop_or_pad(chip, self.target_hw[0], self.target_hw[1])
                    chip2d = chip.reshape(self.C, -1)
                    stack_mean += chip2d.mean(axis=1)
                    stack_var += chip2d.var(axis=1)
                    n += 1
                mean = (stack_mean / max(1, n)).astype(np.float32)
                std = np.sqrt(stack_var / max(1, n) + 1e-8).astype(np.float32)
                self.band_norm = {"mean": mean, "std": std}
        
        self.y = np.array([LABEL_MAP[c] for c in self.table["class"].tolist()], dtype=np.int64)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        chip = read_bsq_chip(Path(row["bsq"]))  # (C, h, w)
    
        # 1) fix channels
        if self.force_k is not None:
            if chip.shape[0] >= self.C:
                chip = chip[:self.C, :, :]
            else:
                pad = np.zeros((self.C - chip.shape[0], chip.shape[1], chip.shape[2]), dtype=chip.dtype)
                chip = np.concatenate([chip, pad], axis=0)
    
        # 2) spatial normalize BEFORE aug (gives consistent crop region)
        if self.target_hw is not None:
            chip = center_crop_or_pad(chip, self.target_hw[0], self.target_hw[1])
    
        # 3) augmentation (may swap H/W)
        if self.augment:
            if random.random() < 0.5:
                chip = chip[:, :, ::-1].copy()
            if random.random() < 0.5:
                chip = chip[:, ::-1, :].copy()
            k = random.randint(0, 3)   # allows 90/270° rotations
            if k:
                chip = np.rot90(chip, k=k, axes=(1, 2)).copy()
            # 4) IMPORTANT: normalize spatial size AGAIN to undo any swap
            if self.target_hw is not None:
                chip = center_crop_or_pad(chip, self.target_hw[0], self.target_hw[1])
    
        # 5) per-band standardization
        if self.band_norm is not None:
            mean = self.band_norm["mean"][:, None, None]
            std = self.band_norm["std"][:, None, None]
            chip = (chip - mean) / (std + 1e-6)
    
        x = torch.from_numpy(chip)  # (C, H, W)
        y = int(LABEL_MAP[row["class"]])
        
        return x, y
