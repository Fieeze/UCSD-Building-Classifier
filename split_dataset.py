"""Split Dataset/raw/<Building>/ into train/ and test/ folders.

    python split_dataset.py

Rerun it whenever you add a new building. It rebuilds train/ and test/ from
raw/, so raw/ stays the single source of truth.

Near-identical photos are grouped and kept on the same side of the split. Two
copies of one photo landing in both train and test would let the model score
well by memorizing rather than learning.
"""

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

RAW_DIR = Path("Dataset/raw")
TRAIN_DIR = Path("Dataset/train")
TEST_DIR = Path("Dataset/test")

TEST_COUNT = 10         # fixed test images per building, not a fraction
HASH_THRESHOLD = 5      # Hamming distance counted as "the same photo"
SEED = 42


def dhash(path, size=8):
    """Perceptual hash — compares each pixel to its right neighbour."""
    with Image.open(path) as img:
        px = np.asarray(img.convert("L").resize((size + 1, size),
                                                Image.Resampling.LANCZOS), dtype=np.int16)
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | int(px[row, col] > px[row, col + 1])
    return bits


def group_duplicates(files):
    """Cluster near-identical photos. Returns a list of lists.

    Compares against every member of a group, not just the first, and merges
    groups that a new photo links together. Duplicate similarity is not
    transitive — A can match B and B match C while A and C do not — so
    comparing against one representative per group lets pairs slip through.
    """
    hashes = {f: dhash(f) for f in files}
    groups = []

    for f in files:
        matched = [g for g in groups
                   if any(bin(hashes[m] ^ hashes[f]).count("1") <= HASH_THRESHOLD
                          for m in g)]
        if not matched:
            groups.append([f])
        else:
            merged = [f]
            for g in matched:
                merged.extend(g)
                groups.remove(g)
            groups.append(merged)

    return groups


def main():
    if not RAW_DIR.is_dir():
        raise SystemExit(f"No {RAW_DIR}/. Put photos in {RAW_DIR}/<Building>/ first.")

    # Rebuild from scratch so reruns are not additive.
    for folder in (TRAIN_DIR, TEST_DIR):
        shutil.rmtree(folder, ignore_errors=True)

    rng = np.random.default_rng(SEED)
    buildings = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())

    print(f"{'Building':<24}{'raw':>6}{'train':>7}{'test':>6}{'dupes':>7}")
    print("-" * 50)

    for building in buildings:
        files = sorted(f for f in building.iterdir()
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not files:
            continue

        groups = group_duplicates(files)
        rng.shuffle(groups)

        # Fill test/ group by group until it reaches TEST_COUNT.
        target = TEST_COUNT
        test_files, train_files = [], []
        for group in groups:
            if len(test_files) < target:
                test_files.extend(group)
            else:
                train_files.extend(group)

        for split_dir, chosen in ((TRAIN_DIR, train_files), (TEST_DIR, test_files)):
            dest = split_dir / building.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in chosen:
                shutil.copy2(f, dest / f.name)

        dupes = len(files) - len(groups)
        print(f"{building.name:<24}{len(files):>6}{len(train_files):>7}"
              f"{len(test_files):>6}{dupes:>7}")

    # On an iCloud-synced Desktop, deleting and recreating train/test right
    # before writing into them can race the sync daemon, which drops in
    # "<Building> 2" conflict-copy folders. They are never real classes, so
    # sweep anything that isn't one of this run's buildings.
    expected = {b.name for b in buildings}
    stray = 0
    for split_dir in (TRAIN_DIR, TEST_DIR):
        for child in split_dir.iterdir():
            if child.is_dir() and child.name not in expected:
                shutil.rmtree(child)
                stray += 1
    if stray:
        print(f"(removed {stray} stray sync-conflict folder(s))")

    print("-" * 50)
    print(f"train -> {TRAIN_DIR}/<Building>/")
    print(f"test  -> {TEST_DIR}/<Building>/")


if __name__ == "__main__":
    main()
