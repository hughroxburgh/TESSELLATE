import cv2
import shutil
import pandas as pd
from pathlib import Path
import numpy as np

# import tkinter as tk
# root = tk.Tk()
# SCREEN_W = root.winfo_screenwidth()
# SCREEN_H = root.winfo_screenheight()
# root.destroy()

print('\n')
sector = input('Which Sector? ')
print('\n')

from screeninfo import get_monitors
monitor = get_monitors()[0]
SCREEN_W, SCREEN_H = monitor.width, monitor.height

# ---- CONFIG ----
IMAGE_DIR = Path(f"/fred/oz335/hroxburg/dev/final_localisation/images")
EVENTS_CSV = Path(f"/fred/oz335/hroxburg/dev/final_localisation/non_flares.csv")
GROUPS = {
    ord("1"): "Junk",
    ord("2"): "CosmicRay",
    ord("3"): "Flare",
    ord("4"): "Asteroid",
    ord("5"): "Variable",
    ord("6"): "Interesting",
}
WINDOW_NAME  = "Image Sorter"
CONTROLS_W   = 440    # ← approximate width of the controls window
CONTROLS_H   = 260    # ← approximate height of the controls window
IMAGE_X,    IMAGE_Y    = 0, 0
CONTROLS_X, CONTROLS_Y = 0, SCREEN_H-CONTROLS_H
# ----------------

df = pd.read_csv(EVENTS_CSV)

group_csv_paths = {}
group_dfs = {}
for group in GROUPS.values():
    folder = IMAGE_DIR / group
    folder.mkdir(exist_ok=True)
    csv_path = folder / "events.csv"
    group_csv_paths[group] = csv_path
    if csv_path.exists():
        group_dfs[group] = pd.read_csv(csv_path)
    else:
        group_dfs[group] = pd.DataFrame(columns=df.columns)

def parse_image_name(stem):
    import re
    m = re.match(r"S(\d+)C(\d+)C(\d+)C(\d+)O(\d+)E(\d+)", stem, re.IGNORECASE)
    if not m:
        return None
    return {
        "sector":  int(m.group(1)),
        "camera":  int(m.group(2)),
        "ccd":     int(m.group(3)),
        "cut":     int(m.group(4)),
        "objid":   int(m.group(5)),
        "eventid": int(m.group(6)),
    }

png_files = sorted(IMAGE_DIR.glob("*.png"))

undo_stack = []

def make_guide_window():
    col1_lines = [
        "  CONTROLS  ",
        "1 : Junk",
        "2 : Cosmic Ray",
        "3 : Flare",
        "4 : Asteroid",
        "5 : Variable",
        "6 : Interesting",
    ]
    col2_lines = [
        "  NAVIGATION  ",
        "Enter  :  Skip",
        "Bkspc  :  Undo",
        "Esc    :  Quit",
    ]

    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.65
    thickness = 1
    pad       = 14

    sizes1 = [cv2.getTextSize(l, font, scale, thickness)[0] for l in col1_lines]
    sizes2 = [cv2.getTextSize(l, font, scale, thickness)[0] for l in col2_lines]

    col1_w = max(s[0] for s in sizes1) + pad * 2
    col2_w = max(s[0] for s in sizes2) + pad * 2
    total_w = col1_w + col2_w

    row_h = max(max(s[1] for s in sizes1), max(s[1] for s in sizes2)) + pad
    total_h = max(len(col1_lines), len(col2_lines)) * row_h + pad

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:] = (0, 0, 139)  # dark red

    # Draw dividing line between columns
    cv2.line(canvas, (col1_w, 0), (col1_w, total_h), (80, 80, 80), 1)

    # Column 1
    y = pad
    for line, (tw, th) in zip(col1_lines, sizes1):
        color = (100, 220, 100) if line.startswith("  CONTROLS") else (220, 220, 220)
        cv2.putText(canvas, line, (pad, y + th), font, scale, color, thickness)
        y += th + pad

    # Column 2
    y = pad
    for line, (tw, th) in zip(col2_lines, sizes2):
        color = (100, 220, 100) if line.startswith("  NAVIGATION") else (220, 220, 220)
        cv2.putText(canvas, line, (col1_w + pad, y + th), font, scale, color, thickness)
        y += th + pad

    return canvas

guide = make_guide_window()
cv2.imshow("Controls", guide)
cv2.moveWindow("Controls", CONTROLS_X, CONTROLS_Y)
cv2.waitKey(1)

i = 0
while i < len(png_files):
    img_path = png_files[i]

    located = None
    if img_path.exists():
        located = img_path
    else:
        for group in GROUPS.values():
            candidate = IMAGE_DIR / group / img_path.name
            if candidate.exists():
                located = candidate
                break

    if located is None:
        print(f"Could not find {img_path.name}, skipping.")
        i += 1
        continue

    img = cv2.imread(str(located))
    if img is None:
        print(f"Could not read {img_path.name}, skipping.")
        i += 1
        continue

    # Resize to fit screen
    max_display_width = SCREEN_W*0.8   # tweak to taste
    if img.shape[1] > max_display_width:
        scale = max_display_width / img.shape[1]
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

    cv2.imshow(WINDOW_NAME, img)
    cv2.moveWindow(WINDOW_NAME, IMAGE_X, IMAGE_Y)
    print(f"[{i+1}/{len(png_files)}] Viewing: {img_path.name}")
    print("Press 1–6 to sort, Enter to skip, Backspace to undo.\n")

    while True:
        key = cv2.waitKey(0)

        # ---- Undo ----
        if key in (8, 127):
            if not undo_stack:
                print("  Nothing to undo.\n")
                continue

            last = undo_stack.pop()
            prev_path    = last["img_path"]
            prev_group   = last["group"]
            prev_matched = last["matched"]

            if prev_group is not None:
                moved_path = IMAGE_DIR / prev_group / prev_path.name
                if moved_path.exists():
                    shutil.move(str(moved_path), str(IMAGE_DIR / prev_path.name))

                if not prev_matched.empty:
                    merged = group_dfs[prev_group].merge(
                        prev_matched, indicator=True, how='left'
                    )
                    group_dfs[prev_group] = merged[merged['_merge'] == 'left_only'].drop(
                        columns='_merge'
                    ).reset_index(drop=True)
                    group_dfs[prev_group].to_csv(group_csv_paths[prev_group], index=False)
                    df = pd.concat([df, prev_matched], ignore_index=True)

            i -= 1
            print(f"  Undid: {prev_path.name} ← back from {prev_group}\n")
            break

        # ---- Quit ----
        if key in (ord('q'), 27):  # q or Escape
            print("Quitting.")
            cv2.destroyAllWindows()
            exit()

        # ---- Skip ----
        if key == 13:
            undo_stack.append({"img_path": img_path, "group": None, "matched": pd.DataFrame()})
            print("Skipped\n")
            i += 1
            break

        # ---- Sort ----
        if key in GROUPS:
            group = GROUPS[key]
            target_dir = IMAGE_DIR / group

            overlay = img.copy()
            text = f"Moved to {group}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 2
            thickness = 3
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            cx = (overlay.shape[1] - tw) // 2
            cy = (overlay.shape[0] + th) // 2
            cv2.rectangle(overlay, (cx-10, cy-th-10), (cx+tw+10, cy+10), (0,0,255), -1)
            cv2.putText(overlay, text, (cx, cy), font, scale, (255,255,255), thickness)
            cv2.imshow(WINDOW_NAME, overlay)
            cv2.moveWindow(WINDOW_NAME, IMAGE_X, IMAGE_Y)
            cv2.waitKey(500)

            shutil.move(str(located), str(target_dir / img_path.name))

            parsed = parse_image_name(img_path.stem)
            matched = pd.DataFrame()
            if parsed:
                mask = (
                    (df["sector"]  == parsed["sector"])  &
                    (df["camera"]  == parsed["camera"])  &
                    (df["ccd"]     == parsed["ccd"])     &
                    (df["cut"]     == parsed["cut"])     &
                    (df["objid"]   == parsed["objid"])   &
                    (df["eventid"] == parsed["eventid"])
                )
                matched = df[mask]
                if matched.empty:
                    print(f"  Warning: no CSV row found for {img_path.name}")
                else:
                    group_dfs[group] = pd.concat(
                        [group_dfs[group], matched], ignore_index=True
                    )
                    group_dfs[group].to_csv(group_csv_paths[group], index=False)
                    df = df[~mask]
            else:
                print(f"  Warning: couldn't parse filename {img_path.name}")

            undo_stack.append({"img_path": img_path, "group": group, "matched": matched})
            print(f"  Moved to {group}\n")
            i += 1
            break

cv2.destroyAllWindows()
