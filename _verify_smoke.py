# -*- coding: utf-8 -*-
"""
Offline END-TO-END smoke test for get_dateset_keypoint.py (ASCII-only on purpose).
Run:  D:\\PX4PSP\\Python38\\python.exe _verify_smoke.py

It replaces VisionCaptureApi / UE4CtrlAPI with fakes, runs main() for a bounded
number of samples in --headless mode, then inspects what actually landed on disk:

  images/*.jpg, labels/*.txt (YOLO-Pose), keypoints.json (COCO), dataset_pose.yaml

Nothing here touches UE4 or the network, and the fake dataset directory is
created inside this experiment folder and removed afterwards.
"""
import json
import os
import shutil
import sys
import types

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT_DIR = os.path.join(HERE, "_smoke_out")
N_SAMPLES = 8


class FakeVision(object):
    """Minimal stand-in for VisionCaptureApi.VisionCaptureApi."""

    def __init__(self):
        self.hasData = [True]
        self.Img = [np.full((480, 640, 3), 90, np.uint8)]
        self.calls = []

    def jsonLoad(self):
        self.calls.append("jsonLoad")

    def sendReqToUE4(self):
        self.calls.append("sendReqToUE4")
        return True

    def startImgCap(self, flag):
        self.calls.append(("startImgCap", flag))


class FakeUE4(object):
    """Minimal stand-in for UE4CtrlAPI.UE4CtrlAPI."""

    def __init__(self):
        self.pos_cmds = []

    def sendUE4Cmd(self, *a, **k):
        pass

    def waitForMapLoaded(self, *a, **k):
        pass

    def sendUE4Pos(self, *a, **k):
        self.pos_cmds.append((a, k))


fake_vis_mod = types.ModuleType("VisionCaptureApi")
fake_ue_mod = types.ModuleType("UE4CtrlAPI")
fake_vis_mod.VisionCaptureApi = FakeVision
fake_ue_mod.UE4CtrlAPI = FakeUE4
sys.modules["VisionCaptureApi"] = fake_vis_mod
sys.modules["UE4CtrlAPI"] = fake_ue_mod

import get_dateset_keypoint as gk  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    return bool(ok)


# ---- no real waiting, deterministic but varied poses ----
import random  # noqa: E402

gk.time.sleep = lambda *a, **k: None
_rng = random.Random(7)
gk.random.randint = lambda a, b: _rng.randint(a, b)   # deterministic pose sequence

if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)

argv = ["get_dateset_keypoint.py",
        "--save-dir", OUT_DIR,
        "--no-timestamp",
        "--headless",
        "--max-samples", str(N_SAMPLES),
        "--min-keypoints", "1"]
sys.argv = argv

print("=" * 72)
print("[1] run main() offline into %s" % os.path.relpath(OUT_DIR, HERE))
print("=" * 72)
rc = gk.main()
check("main() returned 0", rc == 0, "rc=%s" % rc)

img_dir = os.path.join(OUT_DIR, "images")
lab_dir = os.path.join(OUT_DIR, "labels")
coco_path = os.path.join(OUT_DIR, "keypoints.json")
yaml_path = os.path.join(OUT_DIR, "dataset_pose.yaml")

print()
print("=" * 72)
print("[2] files on disk")
print("=" * 72)
imgs = sorted(os.listdir(img_dir)) if os.path.isdir(img_dir) else []
labs = sorted(os.listdir(lab_dir)) if os.path.isdir(lab_dir) else []
check("images/ holds %d jpg" % N_SAMPLES,
      len(imgs) == N_SAMPLES and all(n.endswith(".jpg") for n in imgs),
      ", ".join(imgs))
check("labels/ holds %d txt" % N_SAMPLES, len(labs) == N_SAMPLES, ", ".join(labs))
check("image/label stems match", [n[:-4] for n in imgs] == [n[:-4] for n in labs])
check("keypoints.json written", os.path.isfile(coco_path),
      "%d bytes" % (os.path.getsize(coco_path) if os.path.isfile(coco_path) else -1))
check("dataset_pose.yaml written", os.path.isfile(yaml_path))
check("no .tmp leftovers", not any(n.endswith(".tmp") for n in os.listdir(OUT_DIR)))

with open(coco_path, encoding="utf-8") as f:
    coco = json.load(f)

print()
print("=" * 72)
print("[3] COCO keypoints consistency with labels")
print("=" * 72)
check("images == %d" % N_SAMPLES, len(coco["images"]) == N_SAMPLES,
      "%d" % len(coco["images"]))
check("annotations == %d" % N_SAMPLES, len(coco["annotations"]) == N_SAMPLES,
      "%d" % len(coco["annotations"]))
check("camera block has 640x480 + focal 320",
      coco["camera"]["image_width"] == 640 and coco["camera"]["image_height"] == 480 and
      abs(coco["camera"]["focal_px"] - 320.0) < 1e-6)
check("category has 9 keypoint names", len(coco["categories"][0]["keypoints"]) == 9)
check("single category id 1 used",
      all(a["category_id"] == 1 for a in coco["annotations"]))

by_id = {a["image_id"]: a for a in coco["annotations"]}
ok_link, ok_vis, ok_norm, ok_3d = True, True, True, True
for name in labs:
    stem = int(name[:-4])
    with open(os.path.join(lab_dir, name)) as f:
        toks = f.read().strip().split(" ")
    if len(toks) != 32:
        ok_link = False
        continue
    ann = by_id.get(stem)
    if ann is None:
        ok_link = False
        continue
    label_kp = [float(t) for t in toks[5:]]
    coco_kp = ann["keypoints"]
    visible = 0
    for i in range(9):
        lx, ly, lv = label_kp[3 * i], label_kp[3 * i + 1], int(label_kp[3 * i + 2])
        cx, cy, cv = coco_kp[3 * i], coco_kp[3 * i + 1], int(coco_kp[3 * i + 2])
        if lv != cv:
            ok_vis = False
        if lv == 2:
            visible += 1
            # label coords are normalized, COCO coords are pixels
            if abs(lx * 640 - cx) > 0.01 or abs(ly * 480 - cy) > 0.01:
                ok_norm = False
        if lv == 0 and (cx != 0.0 or cy != 0.0):
            ok_vis = False
    if ann["num_keypoints"] != visible:
        ok_vis = False
    if len(ann.get("keypoints_world_m", [])) != 9 or "target_pose" not in ann:
        ok_3d = False
    for px, py in zip(coco_kp[0::3], coco_kp[1::3]):
        if not (0.0 <= px <= 640.0 and 0.0 <= py <= 480.0):
            ok_norm = False

check("each label links to a COCO annotation (32 fields)", ok_link)
check("visibility codes agree label vs COCO + num_keypoints", ok_vis)
check("normalized label coords == COCO pixel coords", ok_norm)
check("3D world points + target pose stored per sample", ok_3d)

# images referenced by COCO must exist, and image size must match what was saved
img_ok = True
for entry in coco["images"]:
    p = os.path.join(img_dir, entry["file_name"])
    if not os.path.isfile(p):
        img_ok = False
        continue
    im = cv2.imread(p)
    if im is None or im.shape[1] != entry["width"] or im.shape[0] != entry["height"]:
        img_ok = False
check("every COCO image exists on disk with matching size", img_ok)

# labels must be parseable by a YOLO-pose loader (5 + 3*kpt_shape fields, in range)
yolo_ok = True
for name in labs:
    with open(os.path.join(lab_dir, name)) as f:
        for row in f.read().strip().splitlines():
            vals = row.split()
            if len(vals) != 5 + 3 * gk.KEYPOINT_COUNT:
                yolo_ok = False
                continue
            nums = [float(v) for v in vals[1:]]
            coords = nums[0:4] + nums[4::3] + nums[5::3]   # box + all kp x/y, no vis column
            if not all(0.0 <= v <= 1.0 for v in coords):
                yolo_ok = False
            if any(int(nums[4 + 3 * i + 2]) not in (0, 1, 2) for i in range(gk.KEYPOINT_COUNT)):
                yolo_ok = False
check("all label rows are valid YOLO-Pose rows", yolo_ok)

# the yaml must point at the directory we just filled
yaml_text = open(yaml_path, encoding="utf-8").read()
check("yaml path points at the smoke dir",
      OUT_DIR.replace("\\", "/") in yaml_text and "kpt_shape: [9, 3]" in yaml_text)

# image content check: the fake frame is a flat gray, saved untouched (no overlay)
im = cv2.imread(os.path.join(img_dir, imgs[0]))
check("saved image is the raw frame (no drawn overlay)",
      im is not None and int(im.std()) == 0, "std=%s" % (int(im.std()) if im is not None else -1))

print()
print("=" * 72)
print("[4] cleanup")
print("=" * 72)
shutil.rmtree(OUT_DIR, ignore_errors=True)
check("smoke output dir removed", not os.path.isdir(OUT_DIR))

n_fail = sum(1 for _, ok in results if not ok)
print()
print("=" * 72)
print("RESULT: %d checks, %d failed -> %s"
      % (len(results), n_fail, "PASS" if n_fail == 0 else "FAIL"))
print("=" * 72)
sys.exit(1 if n_fail else 0)
