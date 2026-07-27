"""Keyboard-driven labeling tool for the UCSD building dataset.

Shows one photo at a time. Press the hotkey for a building and the file is
moved into that building's folder and the next photo appears. One keypress per
photo, no mouse.

    python label_images.py

Workflow:
    1. Dump every unsorted photo into  Dataset/unsorted/
    2. List your buildings in         Dataset/buildings.txt  (one per line;
                                      created for you on first run)
    3. Run this tool. Labeled photos land in Dataset/labeled/<Building>/
    4. Split into train/val/test afterwards — see labels.csv, which records the
       EXIF capture time of every photo so you can split by session.

Keys:
    1 2 3 … q w e …   assign to the building shown in the legend
    space             skip (leaves the file in unsorted for later)
    u                 undo the last assignment
    x                 reject — move to Dataset/rejected/ (blurry, wrong subject)
    left / right      go back / forward without labeling
    Escape            quit (progress is saved continuously)

Requires tkinter. It ships with python.org and Xcode Python builds; on Homebrew
Python you may need:  brew install python-tk
"""

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    import tkinter as tk
except ImportError:
    sys.exit("tkinter is not available. On Homebrew Python: brew install python-tk")

from PIL import Image, ImageOps, ImageTk

# iPhones shoot HEIC by default, which Pillow cannot open unaided.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}

# 'u' and 'x' are reserved for undo/reject, so they are excluded here.
HOTKEY_POOL = [c for c in "1234567890qwertyiopasdfghjklzcvbnm" if c not in {"u", "x"}]

BUILDINGS_TEMPLATE = """\
# One building per line. The name becomes the folder name, so use underscores
# instead of spaces. Lines starting with # are ignored.
#
# Order matters: the first building gets hotkey 1, the second 2, and so on.
# Put the buildings you photographed most near the top.
#
# Aim for 10-15 buildings. Replace these examples with your own list.

Geisel_Library
Price_Center
Atkinson_Hall
Jacobs_Hall
RIMAC
Warren_Lecture_Hall
Center_Hall
Galbraith_Hall
York_Hall
Pepper_Canyon_Hall
"""


def parse_args():
    p = argparse.ArgumentParser(description="Label building photos with one keypress")
    p.add_argument("--source", default="Dataset/unsorted",
                   help="folder of unlabeled photos (default: Dataset/unsorted)")
    p.add_argument("--dest", default="Dataset/labeled",
                   help="where labeled folders are created (default: Dataset/labeled)")
    p.add_argument("--buildings", default="Dataset/buildings.txt",
                   help="text file listing building names")
    p.add_argument("--copy", action="store_true",
                   help="copy instead of move, leaving the source folder untouched")
    return p.parse_args()


def load_buildings(path: Path) -> "list[str]":
    """Read the building list, creating a template on first run."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(BUILDINGS_TEMPLATE)
        sys.exit(f"Created {path}\nEdit it with your building names, then run this again.")

    names = [line.strip() for line in path.read_text().splitlines()]
    names = [n for n in names if n and not n.startswith("#")]

    if not names:
        sys.exit(f"{path} has no building names in it.")
    if len(names) > len(HOTKEY_POOL):
        sys.exit(f"{len(names)} buildings but only {len(HOTKEY_POOL)} hotkeys available.")
    if len(set(names)) != len(names):
        sys.exit(f"{path} contains duplicate building names.")
    return names


def find_images(source: Path) -> "list[Path]":
    """Every image in the source folder, sorted so runs are reproducible."""
    if not source.is_dir():
        sys.exit(f"No such folder: {source}\nPut your unsorted photos there first.")
    files = sorted(f for f in source.iterdir()
                   if f.is_file() and f.suffix.lower() in IMAGE_EXTS)

    heic = [f for f in files if f.suffix.lower() in {".heic", ".heif"}]
    if heic and not HEIC_SUPPORTED:
        print(f"Warning: {len(heic)} HEIC files found but pillow-heif is not installed.")
        print("         Run 'pip install pillow-heif' to label them.")
        files = [f for f in files if f.suffix.lower() not in {".heic", ".heif"}]
    return files


def read_capture_info(path: Path) -> "tuple[str, str]":
    """(capture timestamp, camera model) from EXIF, or ('', '') if absent.

    Saved to labels.csv because it lets you split train/val/test by capture
    session later — photos from one walk-around must not straddle the split.
    """
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            camera = str(exif.get(272, "")).strip()          # 272 = Model
            # DateTimeOriginal (36867) lives in the Exif sub-IFD, not the base one.
            taken = str(exif.get_ifd(0x8769).get(36867, "")).strip()
            if not taken:
                taken = str(exif.get(306, "")).strip()        # 306 = DateTime
        return taken, camera
    except Exception:
        return "", ""


def unique_destination(folder: Path, filename: str) -> Path:
    """A non-colliding path inside folder, appending _1, _2 … if needed."""
    candidate = folder / filename
    if not candidate.exists():
        return candidate

    stem, suffix = Path(filename).stem, Path(filename).suffix
    n = 1
    while (folder / f"{stem}_{n}{suffix}").exists():
        n += 1
    return folder / f"{stem}_{n}{suffix}"


class LabelApp:
    """Tk window: image on top, hotkey legend below, status bar at the bottom."""

    def __init__(self, root, files, buildings, dest_root, log_path, copy_mode):
        self.root = root
        self.files = files
        self.buildings = buildings
        self.dest_root = dest_root
        self.log_path = log_path
        self.copy_mode = copy_mode

        self.hotkeys = {HOTKEY_POOL[i]: name for i, name in enumerate(buildings)}
        self.index = 0
        self.history = []                              # (src, dest) for undo
        self.counts = {name: 0 for name in buildings}
        self.counts["[rejected]"] = 0
        self._photo = None                             # anchor against GC

        self._count_existing()
        self._build_ui()
        self.root.bind("<Key>", self.on_key)
        self.show_current()

    # ---------- setup ----------

    def _count_existing(self):
        """Seed the per-building counters from folders already on disk."""
        for name in self.buildings:
            folder = self.dest_root / name
            if folder.is_dir():
                self.counts[name] = sum(
                    1 for f in folder.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                )

    def _build_ui(self):
        self.root.title("UCSD Building Labeler")
        self.root.geometry("1100x850")
        self.root.configure(bg="#1e1e1e")

        self.status = tk.Label(self.root, text="", bg="#1e1e1e", fg="#dddddd",
                               font=("Helvetica", 13), anchor="w", padx=12, pady=6)
        self.status.pack(fill="x")

        self.canvas = tk.Label(self.root, bg="#1e1e1e")
        self.canvas.pack(fill="both", expand=True)

        legend = tk.Frame(self.root, bg="#252525")
        legend.pack(fill="x")

        # Legend in columns of 5 so 15 buildings stay on three tidy rows.
        self.legend_labels = {}
        for i, name in enumerate(self.buildings):
            key = HOTKEY_POOL[i]
            lbl = tk.Label(legend, text="", bg="#252525", fg="#cccccc",
                           font=("Menlo", 12), anchor="w", padx=10, pady=2)
            lbl.grid(row=i % 5, column=i // 5, sticky="w")
            self.legend_labels[name] = (lbl, key)

        help_text = ("space skip    u undo    x reject    "
                     "←/→ move    Esc quit")
        tk.Label(self.root, text=help_text, bg="#252525", fg="#888888",
                 font=("Helvetica", 11), pady=6).pack(fill="x")

        self.root.bind("<Configure>", lambda e: None)   # placeholder for resize hooks

    # ---------- rendering ----------

    def show_current(self):
        self._refresh_legend()

        if self.index >= len(self.files):
            self.canvas.configure(image="", text="All done — nothing left to label.",
                                  fg="#8fd694", font=("Helvetica", 22))
            self.status.configure(text=self._summary())
            return

        path = self.files[self.index]
        try:
            with Image.open(path) as img:
                # Phone photos carry an orientation flag; without this the image
                # displays sideways and you mislabel it.
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail((1050, 640), Image.Resampling.LANCZOS)
                self._photo = ImageTk.PhotoImage(img)
            self.canvas.configure(image=self._photo, text="")
        except Exception as exc:
            self._photo = None
            self.canvas.configure(image="", fg="#e06c75", font=("Helvetica", 15),
                                  text=f"Could not open {path.name}\n{exc}")

        self.status.configure(
            text=f"[{self.index + 1}/{len(self.files)}]  {path.name}     {self._summary()}"
        )

    def _refresh_legend(self):
        for name, (lbl, key) in self.legend_labels.items():
            lbl.configure(text=f"[{key}] {name:<24} {self.counts[name]:>4}")

    def _summary(self):
        labeled = sum(self.counts[n] for n in self.buildings)
        thin = [n for n in self.buildings if self.counts[n] < 30]
        note = f"   thin: {len(thin)} building(s) under 30" if thin else ""
        return f"labeled {labeled}{note}"

    # ---------- actions ----------

    def on_key(self, event):
        key = event.keysym

        if key == "Escape":
            self.root.quit()
        elif key == "space":
            self.index += 1
            self.show_current()
        elif key == "Right":
            self.index = min(self.index + 1, len(self.files))
            self.show_current()
        elif key == "Left":
            self.index = max(self.index - 1, 0)
            self.show_current()
        elif event.char == "u":
            self.undo()
        elif event.char == "x":
            self.assign("[rejected]")
        elif event.char in self.hotkeys:
            self.assign(self.hotkeys[event.char])

    def assign(self, label: str):
        if self.index >= len(self.files):
            return

        src = self.files[self.index]
        folder = (self.dest_root.parent / "rejected" if label == "[rejected]"
                  else self.dest_root / label)
        folder.mkdir(parents=True, exist_ok=True)
        dest = unique_destination(folder, src.name)

        if self.copy_mode:
            shutil.copy2(src, dest)
        else:
            shutil.move(str(src), str(dest))

        self.history.append((src, dest))
        self.counts[label] += 1
        self._log(src, label, dest)

        self.index += 1
        self.show_current()

    def undo(self):
        if not self.history:
            return

        src, dest = self.history.pop()
        if self.copy_mode:
            dest.unlink(missing_ok=True)
        else:
            shutil.move(str(dest), str(src))

        # Roll back the counter for whichever folder it went into.
        label = "[rejected]" if dest.parent.name == "rejected" else dest.parent.name
        if label in self.counts:
            self.counts[label] -= 1

        self._drop_last_log_row()
        self.index = max(0, self.index - 1)
        self.show_current()

    # ---------- label log ----------

    def _log(self, src: Path, label: str, dest: Path):
        """Append one row to labels.csv, creating the header on first write."""
        taken, camera = read_capture_info(dest)
        new_file = not self.log_path.exists()

        with self.log_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(["labeled_at", "original_name", "building",
                                 "dest_path", "captured_at", "camera"])
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                src.name,
                label,
                str(dest.relative_to(PROJECT_ROOT)),
                taken,
                camera,
            ])

    def _drop_last_log_row(self):
        """Remove the final CSV row so undo leaves no phantom entry."""
        if not self.log_path.exists():
            return
        rows = self.log_path.read_text().splitlines()
        if len(rows) <= 1:
            self.log_path.unlink()
        else:
            self.log_path.write_text("\n".join(rows[:-1]) + "\n")


def main():
    args = parse_args()

    buildings_path = PROJECT_ROOT / args.buildings
    source = PROJECT_ROOT / args.source
    dest_root = PROJECT_ROOT / args.dest
    log_path = PROJECT_ROOT / "Dataset" / "labels.csv"

    buildings = load_buildings(buildings_path)
    files = find_images(source)

    if not files:
        sys.exit(f"No images in {source}. Put your unsorted photos there first.")

    print(f"{len(files)} images to label across {len(buildings)} buildings.")
    print(f"Labeled files go to {dest_root}/<Building>/")
    print(f"Log: {log_path}\n")

    root = tk.Tk()
    LabelApp(root, files, buildings, dest_root, log_path, args.copy)
    root.mainloop()

    print("Session ended. Re-run any time to continue where you left off.")


if __name__ == "__main__":
    main()
