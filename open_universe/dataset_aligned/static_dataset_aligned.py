# SPDX‑License‑Identifier: Apache‑2.0
"""
Random word‑aligned windows on‑the‑fly, using pre‑computed TextGrids.
Template follows the original NoisyDataset style (to_absolute_path, logging).
"""
from __future__ import annotations
import math, random, logging, os
from pathlib import Path
from typing import Tuple, List, Union
import textgrid, torch, torchaudio
from hydra.utils import to_absolute_path

log = logging.getLogger(__name__)


class WordCutDataset(torch.utils.data.Dataset):
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        audio_root: Union[str, Path],
        textgrid_root: Union[str, Path],
        *,
        split: str = "train",
        fs: int = 16_000,
        win_len_sec: float | None = 4.0,      # None → keep full file
        min_cut_sec: float = 0.5,
        max_cut_sec: float = 2.0,
        num_samples_per_audio: int = 2,
        repeat_sample: bool = False,
        part_used: float = 1.0,
    ):
        super().__init__()

        self.fs = fs
        self.K  = num_samples_per_audio if split == "train" else 1
        self.do_cut = split == "train" and win_len_sec is not None

        self.win_N = int(win_len_sec * fs) if win_len_sec else None
        self.min_N = int(min_cut_sec * fs) if win_len_sec else None
        self.max_N = int(max_cut_sec * fs) if win_len_sec else None
        self.repeat = repeat_sample

        root = Path(to_absolute_path(str(audio_root))) / split
        self.clean = root / "clean"
        self.noisy = root / "noisy"
        # self.tgdir = Path(to_absolute_path(str(textgrid_root))) / split / "align_textgrids"
        self.tgdir = Path(to_absolute_path(str(textgrid_root))) / split

        # --------------------------- file list ---------------------------
        exts = ("*.wav", "*.flac", "*.ogg", "*.mp3")
        self.files: List[Path] = []
        for pat in exts:
            self.files.extend(self.clean.glob(pat))
        self.files = sorted(self.files)
        if part_used < 1.0:
            self.files = self.files[: max(1, int(len(self.files) * part_used))]

        if not self.files:
            raise ValueError(
                f"No audio files found in {self.clean} (checked {', '.join(exts)})"
            )

        # --------------------------- cache intervals --------------------
        self.intervals: List[List[Tuple[int, int, str]]] = []
            
        keep_files, keep_iv = [], []
        
        for wav in self.files:
            tg_path = self.tgdir / f"{wav.stem}.TextGrid"
            if not tg_path.exists():
                print(f"[dbg] No TextGrid for {wav.name} – skipped.")
                continue
            tg = textgrid.TextGrid.fromFile(str(tg_path))
            
            tier = next(t for t in tg.tiers if "word" in t.name.lower())
            iv = [
                (int(w.minTime * fs), int(w.maxTime * fs), w.mark.strip())
                for w in tier.intervals
                if w.mark.strip()
            ]

            keep_files.append(wav)
            keep_iv.append(iv)

        self.files      = keep_files
        self.intervals  = keep_iv
        log.info(f"[{split}] {len(self.files)} wavs × {self.K} samples "
                 f"(TextGrid missing for {len(exts)-len(self.files)} files)")
        
        print(f"[{split}] {len(self.files)} wavs × {self.K} samples "
                 f"(TextGrid missing for {len(exts)-len(self.files)} files)")


    # ------------------------------------------------------------------ #
    def __len__(self): return len(self.files) * self.K

    # ------------------------- helpers --------------------------------- #
    def _choose_cut(self, iv, wav_N):
        bounds = {0, wav_N}
        for s, e, _ in iv:
            bounds.update([s, e])
        bounds = sorted(bounds)
        valid = [(s, e) for s in bounds for e in bounds
                 if self.min_N <= e - s <= self.max_N and e > s]
        return random.choice(valid) if valid else None

    def _tile_or_pad(self, clip: torch.Tensor):
        if self.win_N is None:                     # keep full length (val/test)
            return clip, 0
        if clip.shape[-1] == 0:
            pad = self.win_N
            return torch.zeros_like(clip.expand(1, self.win_N)), pad
        if self.repeat:
            reps = math.ceil(self.win_N / clip.shape[-1])
            clip = clip.repeat(1, reps)
        pad = self.win_N - clip.shape[-1]
        if pad > 0:
            clip = torch.nn.functional.pad(clip, (0, pad))
        return clip[..., : self.win_N], pad

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        fi   = idx // self.K
        wavN = self.files[fi]

        wav_c, _ = torchaudio.load(wavN)
        wav_n, _ = torchaudio.load(self.noisy / wavN.name)
        wav_N    = wav_c.shape[-1]

        if self.do_cut:
            cut = self._choose_cut(self.intervals[fi], wav_N)
            if cut is None:
                start, end, words = 0, wav_N, []
            else:
                s, e = cut
                start, end = s, e
                words = [w for s_, e_, w in self.intervals[fi] if s_ >= s and e_ <= e]
        else:
            start, end, words = 0, wav_N, [w for _, _, w in self.intervals[fi]]

        clip_c, pad = self._tile_or_pad(wav_c[:, start:end])
        clip_n, _   = self._tile_or_pad(wav_n[:, start:end])

        if self.win_N:
            mask = torch.ones(self.win_N); mask[-pad:] = 0
        else:
            mask = torch.ones(clip_n.shape[-1])

        txt = " ".join(words).lower().strip()

        # debug first sample
        if idx < 2:
            print(f"[dbg] {wavN.name} txt='{txt[:60]}' pad={pad}")

        return clip_n, clip_c, txt, mask
