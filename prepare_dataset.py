"""Turn labeled photos into a k-fold cross-validation manifest.

Reads  Dataset/labeled/<Building>/*.jpg  and writes  Dataset/splits.csv,
assigning every photo a fold number. Nothing is copied or moved — training
reads the manifest directly, so there is exactly one copy of each image on disk.

    python prepare_dataset.py
    python prepare_dataset.py --folds 5 --drop-duplicates
    python prepare_dataset.py --holdout-test 0.15    # optional untouched test set

Three things happen, in order:

1. SESSION GROUPING. Photos are clustered by EXIF capture time: a gap longer
   than --session-gap minutes starts a new session. A session is one
   walk-around of one building, and every photo in it is near-identical to its
   neighbours. Sessions are kept whole inside a fold, so a photo and its
   near-twin can never end up on opposite sides of the split. Without this,
   accuracy looks great and means nothing.

2. DUPLICATE DETECTION. A perceptual hash (dHash) finds near-identical photos
   within each building. They are flagged, and with --drop-duplicates excluded.

3. FOLD ASSIGNMENT. StratifiedGroupKFold splits by session while keeping each
   building's proportion roughly equal in every fold.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="Build a k-fold manifest from labeled photos")
    p.add_argument("--labeled", default="Dataset/labeled",
                   help="folder of <Building>/ subfolders (default: Dataset/labeled)")
    p.add_argument("--out", default="Dataset/splits.csv")
    p.add_argument("--folds", type=int, default=5,
                   help="number of CV folds (default: 5)")
    p.add_argument("--session-gap", type=int, default=30, metavar="MIN",
                   help="minutes between photos that starts a new session (default: 30)")
    p.add_argument("--hash-threshold", type=int, default=5,
                   help="max Hamming distance counted as a near-duplicate (default: 5)")
    p.add_argument("--min-detail", type=float, default=3.0,
                   help="images below this detail score are too featureless to "
                        "hash reliably and are reported, not deduplicated (default: 3). "
                        "Reference points: flat wall 0, screenshot 9, sky gradient 29, "
                        "ordinary photo 30-60")
    p.add_argument("--drop-duplicates", action="store_true",
                   help="exclude flagged duplicates from the manifest entirely")
    p.add_argument("--holdout-test", type=float, default=0.0, metavar="FRAC",
                   help="hold out this fraction of sessions as an untouched test set")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------
# 1. Capture times and sessions
# --------------------------------------------------------------------------

def read_capture_time(path: Path):
    """EXIF capture datetime, falling back to file mtime.

    Deliberately duplicated from label_images.py so that each tool stays
    runnable on its own — labeling should not require sklearn installed.
    """
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            raw = str(exif.get_ifd(0x8769).get(36867, "")).strip()   # DateTimeOriginal
            if not raw:
                raw = str(exif.get(306, "")).strip()                 # DateTime
        if raw:
            return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S"), True
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime), False


def assign_sessions(records, gap_minutes: int) -> int:
    """Cluster each building's photos into time-contiguous sessions.

    Mutates records in place, setting record['session']. Returns the number of
    photos whose time came from mtime rather than EXIF.
    """
    by_building = defaultdict(list)
    for rec in records:
        by_building[rec["building"]].append(rec)

    gap = gap_minutes * 60
    for building, group in by_building.items():
        group.sort(key=lambda r: r["taken"])
        session_idx = 0
        previous = None
        for rec in group:
            if previous is not None and (rec["taken"] - previous).total_seconds() > gap:
                session_idx += 1
            rec["session"] = f"{building}__s{session_idx}"
            previous = rec["taken"]

    return sum(1 for r in records if not r["exif"])


# --------------------------------------------------------------------------
# 2. Near-duplicate detection
# --------------------------------------------------------------------------

def dhash(path: Path, size: int = 8) -> "tuple[int, float]":
    """(perceptual hash, detail score) for one image.

    dHash compares each pixel to its right neighbour, which makes it robust to
    resizing and mild compression — it catches "the same photo saved twice" and
    consecutive burst frames, which exact checksums miss.

    The detail score is the pixel standard deviation of the same downscaled
    image. It exists because a featureless photo (blank sky, blown-out wall,
    badly out-of-focus shot) has no neighbour differences at all and hashes to
    0 — so every featureless photo looks identical to every other one. Without
    this guard they would all be flagged as duplicates of each other.
    """
    with Image.open(path) as img:
        small = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        pixels = np.asarray(small, dtype=np.int16)

    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | int(pixels[row, col] > pixels[row, col + 1])
    return bits, float(pixels.std())


def flag_duplicates(records, threshold: int, min_detail: float) -> "tuple[int, list]":
    """Mark near-duplicates within each building; keep the first of each cluster.

    Compared within a building only: two similar photos of the SAME building are
    redundant data, whereas similar photos of different buildings are a labeling
    problem, reported separately.

    Returns (flagged count, low-detail records skipped by the comparison).
    """
    low_detail = [r for r in records if r["detail"] < min_detail]
    comparable = defaultdict(list)
    for rec in records:
        if rec["detail"] >= min_detail:
            comparable[rec["building"]].append(rec)

    flagged = 0
    for group in comparable.values():
        kept = []
        for rec in group:
            twin = next(
                (k for k in kept
                 if bin(k["phash"] ^ rec["phash"]).count("1") <= threshold),
                None,
            )
            if twin is None:
                kept.append(rec)
            else:
                rec["duplicate_of"] = twin["path"]
                flagged += 1
    return flagged, low_detail


def find_cross_building_collisions(records, threshold: int, min_detail: float,
                                   limit: int = 50):
    """Identical-looking photos filed under different buildings — likely mislabels.

    Stops at `limit` pairs: this comparison is quadratic, and if there are
    hundreds of hits the cause is systematic (usually low-detail photos), so
    listing every pair helps nobody.
    """
    usable = [r for r in records if r["detail"] >= min_detail]
    collisions = []

    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            if a["building"] == b["building"]:
                continue
            if bin(a["phash"] ^ b["phash"]).count("1") <= threshold:
                collisions.append((a, b))
                if len(collisions) >= limit:
                    return collisions
    return collisions


# --------------------------------------------------------------------------
# 3. Fold assignment
# --------------------------------------------------------------------------

def assign_folds(records, n_folds: int, seed: int):
    """Give every record a fold index via StratifiedGroupKFold.

    groups = session  -> a session lands entirely in one fold
    y      = building -> each fold keeps roughly the same class proportions
    """
    y = np.array([r["label"] for r in records])
    groups = np.array([r["session"] for r in records])
    X = np.zeros(len(records))            # unused by the splitter, but required

    sessions_per_class = Counter()
    for building, sessions in group_sessions_by_class(records).items():
        sessions_per_class[building] = len(sessions)

    too_few = {b: n for b, n in sessions_per_class.items() if n < n_folds}
    if too_few:
        detail = "\n".join(f"    {b}: {n} session(s)" for b, n in sorted(too_few.items()))
        sys.exit(
            f"Cannot build {n_folds} folds — these buildings have fewer capture "
            f"sessions than folds:\n{detail}\n\n"
            f"Each session must sit entirely in one fold, so you need at least "
            f"{n_folds} per building.\nEither photograph them again on another day, "
            f"or lower --folds (e.g. --folds {max(2, min(sessions_per_class.values()))})."
        )

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (_, val_idx) in enumerate(splitter.split(X, y, groups)):
        for i in val_idx:
            records[i]["fold"] = fold_idx


def group_sessions_by_class(records):
    """{building: {session ids}} — used for the fold-count sanity check."""
    out = defaultdict(set)
    for rec in records:
        out[rec["building"]].add(rec["session"])
    return out


def holdout_sessions(records, fraction: float, seed: int):
    """Reserve whole sessions as a test set that CV never touches.

    Optional. With very little data you may prefer to skip this and let the
    cross-validated score be your headline number.
    """
    rng = np.random.default_rng(seed)
    held = set()

    for building, sessions in group_sessions_by_class(records).items():
        ordered = sorted(sessions)
        n = int(round(len(ordered) * fraction))
        if n == 0 or len(ordered) - n < 2:
            continue                      # leave at least 2 sessions for training
        held.update(rng.choice(ordered, size=n, replace=False).tolist())

    for rec in records:
        if rec["session"] in held:
            rec["fold"] = "test"
    return held


# --------------------------------------------------------------------------

def collect_records(labeled_dir: Path):
    """One dict per image, with building, label index and capture time."""
    buildings = sorted(d.name for d in labeled_dir.iterdir()
                       if d.is_dir() and not d.name.startswith("."))
    if not buildings:
        sys.exit(f"No building folders inside {labeled_dir}. Run label_images.py first.")

    label_of = {name: i for i, name in enumerate(buildings)}
    records = []

    for building in buildings:
        for path in sorted((labeled_dir / building).iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            taken, from_exif = read_capture_time(path)
            records.append({
                "path": str(path.relative_to(PROJECT_ROOT)),
                "building": building,
                "label": label_of[building],
                "taken": taken,
                "exif": from_exif,
                "session": "",
                "phash": 0,
                "detail": 0.0,
                "duplicate_of": "",
                "fold": -1,
            })

    if not records:
        sys.exit(f"Found {len(buildings)} building folders but no images in them.")
    return records, buildings


def print_summary(records, buildings, n_folds, holdout):
    """Per-building table plus the class-balance and thin-class warnings."""
    active = [r for r in records if r["fold"] != -1]
    fold_ids = sorted({r["fold"] for r in active if r["fold"] != "test"})

    header = f"{'Building':<26}{'imgs':>6}{'sess':>6}  " + "".join(
        f"{'f' + str(f):>6}" for f in fold_ids)
    if holdout:
        header += f"{'test':>7}"
    print("\n" + header)
    print("-" * len(header))

    sessions = group_sessions_by_class(records)
    for building in buildings:
        rows = [r for r in active if r["building"] == building]
        line = f"{building:<26}{len(rows):>6}{len(sessions[building]):>6}  "
        line += "".join(f"{sum(1 for r in rows if r['fold'] == f):>6}" for f in fold_ids)
        if holdout:
            line += f"{sum(1 for r in rows if r['fold'] == 'test'):>7}"
        print(line)

    counts = [sum(1 for r in active if r["building"] == b) for b in buildings]
    print("-" * len(header))
    print(f"{'TOTAL':<26}{sum(counts):>6}")

    thin = [b for b, c in zip(buildings, counts) if c < 100]
    if thin:
        print(f"\n  Under 100 images: {', '.join(thin)}")
        print("  These will drag down accuracy. Aim for 150-300 per building.")

    if counts and max(counts) > 2 * min(counts):
        print(f"\n  Imbalanced: largest class is {max(counts) / min(counts):.1f}x the smallest.")
        print("  The model will favour the common buildings. Shoot more of the rare ones.")


def main():
    args = parse_args()

    labeled_dir = PROJECT_ROOT / args.labeled
    out_path = PROJECT_ROOT / args.out
    if not labeled_dir.is_dir():
        sys.exit(f"No such folder: {labeled_dir}\nRun label_images.py first.")

    records, buildings = collect_records(labeled_dir)
    print(f"Found {len(records)} images across {len(buildings)} buildings.")

    # 1. sessions
    no_exif = assign_sessions(records, args.session_gap)
    n_sessions = len({r["session"] for r in records})
    print(f"Grouped into {n_sessions} capture sessions "
          f"(gap > {args.session_gap} min starts a new one).")
    if no_exif:
        print(f"  Warning: {no_exif} photo(s) had no EXIF timestamp; used file mtime.")
        print("  Screenshots and re-saved images lose EXIF — sessions may be wrong for those.")

    # 2. duplicates
    print("Hashing images for duplicate detection…")
    for rec in records:
        rec["phash"], rec["detail"] = dhash(PROJECT_ROOT / rec["path"])

    flagged, low_detail = flag_duplicates(records, args.hash_threshold, args.min_detail)
    print(f"Flagged {flagged} near-duplicate(s)"
          + (" — excluded." if args.drop_duplicates else " — kept, see splits.csv."))

    if low_detail:
        print(f"\n  {len(low_detail)} photo(s) are nearly featureless "
              f"(detail < {args.min_detail}) and were skipped by the duplicate check.")
        print("  Usually blank sky, a blown-out wall, or a badly blurred shot. Review them:")
        for rec in sorted(low_detail, key=lambda r: r["detail"])[:5]:
            print(f"    {rec['path']}  (detail {rec['detail']:.1f})")
        if len(low_detail) > 5:
            print(f"    … and {len(low_detail) - 5} more")

    collisions = find_cross_building_collisions(records, args.hash_threshold,
                                                args.min_detail)
    if collisions:
        print(f"\n  {len(collisions)}+ photo pair(s) look identical but are filed under "
              f"different buildings — probably mislabeled:")
        for a, b in collisions[:5]:
            print(f"    {a['path']}  <->  {b['path']}")

    usable = [r for r in records if not (args.drop_duplicates and r["duplicate_of"])]

    # 3. folds
    if args.holdout_test > 0:
        held = holdout_sessions(usable, args.holdout_test, args.seed)
        print(f"\nHeld out {len(held)} session(s) as an untouched test set.")
        cv_records = [r for r in usable if r["fold"] != "test"]
    else:
        cv_records = usable

    assign_folds(cv_records, args.folds, args.seed)

    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "building", "label", "session", "fold",
                         "captured_at", "duplicate_of"])
        for rec in records:
            writer.writerow([
                rec["path"], rec["building"], rec["label"], rec["session"],
                rec["fold"], rec["taken"].isoformat(timespec="seconds"),
                rec["duplicate_of"],
            ])

    print_summary(records, buildings, args.folds, args.holdout_test > 0)
    print(f"\nManifest: {out_path}")
    print(f"Next    : python Model/train.py --cv")


if __name__ == "__main__":
    main()
