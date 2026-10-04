# -*- coding: utf-8 -*-
"""
Offline self-check for get_dateset_keypoint.py (ASCII-only source on purpose).
Run:  D:\\PX4PSP\\Python38\\python.exe _verify_keypoints.py
      (any python3 with numpy + opencv works; no UE4 / RflySim3D needed)

Checks:
  1. get_dateset.py is never modified (sha256 guard, read only)
  2. getUav9Point() == POINT_OFFSETS (body-frame round trip)
  3. projection of the new keypoint function == original get_pic_situation()
     over many random target poses (must match to the last rounded pixel)
  4. visibility coding: 2 inside frame / 1 outside frame / 0 behind camera
  5. YOLO-Pose label line: 31 fields, normalized, vis column consistent
  6. COCO keypoints doc/annotation structure + JSON serializability
  7. dataset_pose.yaml content (kpt_shape / names / flip warning)
  8. render check -> _verify_keypoints.png
No dataset folder is created in this experiment directory (temp dir is used).
"""
import hashlib
import json
import math
import os
import random
import shutil
import sys
import types

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TARGET = "get_dateset_keypoint.py"
PROTECTED = "get_dateset.py"

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    return bool(ok)


# ---- stub the RflySim API modules so importing the recorder reaches no UE4 ----
fake_vis = types.ModuleType("VisionCaptureApi")
fake_vis.VisionCaptureApi = lambda *a, **k: types.SimpleNamespace()
fake_ue = types.ModuleType("UE4CtrlAPI")
fake_ue.UE4CtrlAPI = lambda *a, **k: types.SimpleNamespace()
sys.modules["VisionCaptureApi"] = fake_vis
sys.modules["UE4CtrlAPI"] = fake_ue

with open(os.path.join(HERE, PROTECTED), "rb") as f:
    protected_before = f.read()
print("[guard] %s read-only baseline: %d bytes, sha256=%s"
      % (PROTECTED, len(protected_before),
         hashlib.sha256(protected_before).hexdigest()))

print("=" * 72)
print("[0] import the recorder as a module (must have no side effects)")
print("=" * 72)
import get_dateset_keypoint as gk  # noqa: E402

check("module imports without touching UE4", gk.HAS_RFLYSIM is True,
      "HAS_RFLYSIM=%s (stubs registered)" % gk.HAS_RFLYSIM)
check("keypoint count == 9", gk.KEYPOINT_COUNT == 9, "count=%d" % gk.KEYPOINT_COUNT)
check("names/offsets lengths match",
      len(gk.KEYPOINT_NAMES) == gk.KEYPOINT_COUNT and
      len(gk.POINT_OFFSETS) == gk.KEYPOINT_COUNT)

# ---- rebuild the ORIGINAL script's geometry, read-only, with stubs ----
# utf-8-sig: get_dateset.py starts with a UTF-8 BOM, which would break exec()
with open(os.path.join(HERE, PROTECTED), encoding="utf-8-sig") as f:
    orig_src = f.read()
head = orig_src[:orig_src.index("startTime = time.time()")]
lines = []
for ln in head.splitlines():
    s = ln.strip()
    if s.startswith("import os") or s.startswith("from os"):
        continue
    if s.startswith("ue.sendUE4Cmd") or s.startswith("ue.waitForMapLoaded") or \
       s.startswith("ue.sendUE4Pos") or s.startswith("vis.") or s.startswith("isSuss ="):
        continue
    lines.append(ln)

_guard_os = types.ModuleType("os")
_guard_os.__dict__.update({k: v for k, v in os.__dict__.items() if k != "makedirs"})
_guard_os.makedirs = lambda *a, **k: None   # never create the timestamp folder here
orig_ns = {"__name__": "_geo_probe", "isSuss": True, "os": _guard_os}
exec("\n".join(lines), orig_ns)

orig_getUav9Point = orig_ns["getUav9Point"]
orig_get_pic_situation = orig_ns["get_pic_situation"]
orig_eul2rot = orig_ns["eul2rot"]
orig_copterCenterHeight = orig_ns["copterCenterHeight"]

print()
print("=" * 72)
print("[1] geometry: getUav9Point() vs POINT_OFFSETS")
print("=" * 72)
random.seed(20240516)
PosInit = [0, 0, -8.086 - 4]
angCopterE = [0, 0, 0]
cameraPosForUav = [0.03, 0, 0]
cam_pos = np.array(PosInit) + np.transpose(np.dot(
    np.linalg.inv(orig_eul2rot(angCopterE)), np.transpose(cameraPosForUav)))

max_off_err = 0.0
max_vs_orig = 0.0
for _ in range(400):
    TargePos = [1.03 + random.uniform(0, 2.0), random.uniform(-0.6, 0.6),
                -12.086 + random.uniform(-0.5, 0.5)]
    TargeAng = [random.uniform(-0.35, 0.35), random.uniform(-0.35, 0.35),
                math.pi / 2 + random.uniform(-0.52, 0.52)]
    pts_new = gk.getUav9Point(TargePos, gk.copterCenterHeight, TargeAng)
    pts_old = orig_getUav9Point(TargePos, orig_copterCenterHeight, TargeAng)
    center_world = np.array(TargePos) + [0, 0, -gk.copterCenterHeight]
    R = gk.eul2rot(TargeAng)
    for i, (pt, off) in enumerate(zip(pts_new, gk.POINT_OFFSETS)):
        delta = np.dot(R, (np.array(pt) - center_world))
        max_off_err = max(max_off_err, float(np.max(np.abs(delta - np.array(off)))))
        max_vs_orig = max(max_vs_orig, float(np.max(np.abs(np.array(pt) - np.array(pts_old[i])))))

check("all 9 points map back to POINT_OFFSETS", max_off_err < 1e-9,
      "max err = %.2e" % max_off_err)
check("new getUav9Point() == original getUav9Point()", max_vs_orig < 1e-9,
      "max err = %.2e m" % max_vs_orig)

print()
print("=" * 72)
print("[2] projection: new project_point_pinhole() vs original get_pic_situation()")
print("=" * 72)
n_cmp = 0
n_match = 0
max_du = 0.0
max_dv = 0.0
n_inside = 0
for _ in range(600):
    TargePos = [1.03 + random.uniform(0, 2.0), random.uniform(-0.6, 0.6),
                -12.086 + random.uniform(-0.5, 0.5)]
    TargeAng = [random.uniform(-0.35, 0.35), random.uniform(-0.35, 0.35),
                math.pi / 2 + random.uniform(-0.52, 0.52)]
    for pt in gk.getUav9Point(TargePos, gk.copterCenterHeight, TargeAng):
        rel = np.dot(orig_eul2rot(angCopterE), np.transpose(
            [pt[0] - cam_pos[0], pt[1] - cam_pos[1], pt[2] - cam_pos[2]]))
        if rel[0] <= gk.MIN_DEPTH:
            continue
        ref = orig_get_pic_situation(np.transpose(rel), gk.px, gk.py, gk.focal)
        u, v, d, vis = gk.project_point_pinhole(pt, cam_pos, angCopterE)
        n_cmp += 1
        max_du = max(max_du, abs(round(u) - ref[0]))
        max_dv = max(max_dv, abs(round(v) - ref[1]))
        if round(u) == ref[0] and round(v) == ref[1]:
            n_match += 1
        n_inside += 1 if vis == gk.VIS_VISIBLE else 0

check("pixel-exact match with the original projection", n_match == n_cmp,
      "%d/%d matched, max |du|=%d max |dv|=%d" % (n_match, n_cmp, max_du, max_dv))
check("samples actually fall inside the frame", n_inside > 0,
      "%d of %d points inside %dx%d" % (n_inside, n_cmp, gk.pic_w, gk.pic_h))

print()
print("=" * 72)
print("[3] visibility coding 2/1/0")
print("=" * 72)
origin = [0.0, 0.0, 0.0]
identity = [0.0, 0.0, 0.0]
u, v, d, vis = gk.project_point_pinhole([1.0, 0.0, 0.0], origin, identity)
check("on-axis point -> visible(2)", vis == gk.VIS_VISIBLE and d is not None,
      "u=%.1f v=%.1f vis=%d" % (u, v, vis))
u, v, d, vis = gk.project_point_pinhole([1.0, 5.0, 0.0], origin, identity)
check("far-lateral point -> occluded(1) out of frame",
      vis == gk.VIS_OCCLUDED and u > gk.pic_w, "u=%.1f vis=%d" % (u, vis))
u, v, d, vis = gk.project_point_pinhole([-1.0, 0.0, 0.0], origin, identity)
check("point behind camera -> not labeled(0)",
      vis == gk.VIS_NOT_LABELED and d is None, "u=%.1f v=%.1f vis=%d" % (u, v, vis))
u, v, d, vis = gk.project_point_pinhole([0.01, 0.0, 0.0], origin, identity)
check("depth < MIN_DEPTH -> not labeled(0)", vis == gk.VIS_NOT_LABELED,
      "depth=%s vis=%d" % (d, vis))

print()
print("=" * 72)
print("[4] YOLO-Pose label line format")
print("=" * 72)
kpts_demo = [(100.5, 200.25, gk.VIS_VISIBLE),
             (-30.0, 200.0, gk.VIS_OCCLUDED),
             (0.0, 0.0, gk.VIS_NOT_LABELED)] + [(300.0, 240.0, gk.VIS_VISIBLE)] * 6
bbox_demo = [50.0, 100.0, 400.0, 380.0]
line = gk.build_yolo_pose_line(0, bbox_demo, kpts_demo).strip()
toks = line.split(" ")
check("field count == 5 + 3*9 = 32", len(toks) == 32, "got %d" % len(toks))
check("class id == 0", toks[0] == "0", "got %r" % toks[0])
nums = [float(t) for t in toks[1:]]
check("all values finite", all(math.isfinite(x) for x in nums))
check("box values normalized to [0,1]", all(0.0 <= x <= 1.0 for x in nums[0:4]),
      "box=%s" % nums[0:4])
check("box center/width correct",
      abs(nums[0] - 225.0 / gk.pic_w) < 1e-6 and abs(nums[2] - 350.0 / gk.pic_w) < 1e-6,
      "cx=%.6f w=%.6f" % (nums[0], nums[2]))
kp = nums[4:]
check("keypoint count in line == 9*3", len(kp) == 27, "got %d" % len(kp))
kp_xy = kp[0::3] + kp[1::3]
check("keypoint x/y normalized to [0,1]", all(0.0 <= x <= 1.0 for x in kp_xy),
      "x/y range = [%.3f, %.3f]" % (min(kp_xy), max(kp_xy)))
vis_col = [int(kp[3 * i + 2]) for i in range(9)]
check("vis column == [2,1,0,...]", vis_col == [2, 1, 0, 2, 2, 2, 2, 2, 2], str(vis_col))
check("occluded kp clamped to frame edge",
      abs(kp[3] - 0.0) < 1e-9 and int(kp[5]) == 1, "x=%.6f" % kp[3])
check("invalid kp written as 0 0 0",
      kp[6] == 0.0 and kp[7] == 0.0 and int(kp[8]) == 0, "%s" % kp[6:9])

print()
print("=" * 72)
print("[5] COCO keypoints annotation / document")
print("=" * 72)
world_pts = [np.array([1.0 + i * 0.01, 0.2, -12.0]) for i in range(9)]
ann = gk.build_coco_annotation(1, 7, bbox_demo, kpts_demo, world_points=world_pts,
                               target_pos=[2.03, 0.3, -12.0],
                               target_ang=[0.2, -0.15, math.pi / 2])
check("len(keypoints) == 27", len(ann["keypoints"]) == 27,
      "got %d" % len(ann["keypoints"]))
check("num_keypoints == count(vis==2)", ann["num_keypoints"] == 7,
      "got %d" % ann["num_keypoints"])
check("bbox == [x,y,w,h]", ann["bbox"] == [50.0, 100.0, 350.0, 280.0], str(ann["bbox"]))
check("world coords + pose recorded",
      len(ann["keypoints_world_m"]) == 9 and "target_pose" in ann)
doc = gk.build_coco_doc([{"id": 7, "file_name": "7.jpg", "width": 640, "height": 480}],
                        [ann], gk.build_camera_info(cam_pos, angCopterE))
check("categories expose 9 keypoints + skeleton",
      doc["categories"][0]["keypoints"] == gk.KEYPOINT_NAMES and
      len(doc["categories"][0]["skeleton"]) == 12)
check("keypoint_definition keeps body offsets",
      doc["keypoint_definition"]["offsets_body_m"] == [list(o) for o in gk.POINT_OFFSETS])
tmp_dir = os.path.join(HERE, "_verify_tmp")   # workspace-local: %TEMP% may be read-only
if os.path.isdir(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir)
tmp_json = os.path.join(tmp_dir, "keypoints.json")
try:
    gk.write_json_atomic(tmp_json, doc)
    with open(tmp_json, encoding="utf-8") as f:
        reread = json.load(f)
    ok_json = (reread["annotations"][0]["keypoints"] == ann["keypoints"] and
               len(reread["annotations"][0]["keypoints_world_m"]) == 9)
    err_json = ""
except Exception as exc:      # numpy values or non-serializable objects
    ok_json, err_json = False, repr(exc)
check("document is JSON-serializable (incl. numpy world points)", ok_json, err_json)
check("json file actually written",
      os.path.exists(tmp_json) and os.path.getsize(tmp_json) > 0,
      "%d bytes" % (os.path.getsize(tmp_json) if os.path.exists(tmp_json) else -1))
check("atomic writer leaves no .tmp behind",
      not os.path.exists(tmp_json + ".tmp"))

print()
print("=" * 72)
print("[6] dataset_pose.yaml")
print("=" * 72)
yaml_text = gk.build_dataset_yaml(tmp_dir, "images", 0, "uav")
check("kpt_shape: [9, 3] present", "kpt_shape: [9, 3]" in yaml_text)
check("names/class present", "0: uav" in yaml_text)
check("train/val point at images", "train: images" in yaml_text and "val: images" in yaml_text)
check("flip-augmentation warning present", "fliplr" in yaml_text and "flip_idx" in yaml_text)
check("path uses forward slashes", "\\" not in yaml_text.split("\n")[3])

print()
print("=" * 72)
print("[7] keypoint set symmetry fact check (why flip augmentation is unsafe)")
print("=" * 72)
offs = np.array(gk.POINT_OFFSETS)
mirrored = offs * np.array([1.0, -1.0, 1.0])
n_sym = sum(1 for m in mirrored
            if np.min(np.max(np.abs(offs - m), axis=1)) < 1e-9)
check("set is NOT left-right symmetric (docs say so)", n_sym < len(offs),
      "%d of %d points have a y-mirror partner" % (n_sym, len(offs)))

print()
print("=" * 72)
print("[8] draw_keypoints() render check -> _verify_keypoints.png")
print("=" * 72)
TargePos = [2.03, 0.3, -12.086 + 0.2]
TargeAng = [0.2, -0.15, math.pi / 2 + 0.3]
pts = gk.getUav9Point(TargePos, gk.copterCenterHeight, TargeAng)
kpts, depths = gk.project_keypoints(pts, cam_pos, angCopterE)
used = [(u, v) for (u, v, vis) in kpts if int(vis) >= gk.VIS_OCCLUDED]
bbox = [min(p[0] for p in used), min(p[1] for p in used),
        max(p[0] for p in used), max(p[1] for p in used)]
canvas = np.full((gk.pic_h, gk.pic_w, 3), 60, np.uint8)
canvas = gk.draw_keypoints(canvas, kpts, bbox_xyxy=bbox, detail=True)
png_path = os.path.join(HERE, "_verify_keypoints.png")
cv2.imwrite(png_path, canvas)
non_bg = int(np.count_nonzero(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) != 60))
print("  visible keypoints: %d/9, bbox=%s" % (gk.count_visible(kpts), [round(b) for b in bbox]))
print("  wrote %s (%dx%d), non-background pixels = %d"
      % (os.path.basename(png_path), canvas.shape[1], canvas.shape[0], non_bg))
check("renders something (points/lines/labels)", non_bg > 500, "%d px" % non_bg)
check("visible count consistent with bbox", gk.count_visible(kpts) >= 1,
      "%d visible" % gk.count_visible(kpts))

print()
print("=" * 72)
print("[9] guard: get_dateset.py untouched")
print("=" * 72)
with open(os.path.join(HERE, PROTECTED), "rb") as f:
    protected_after = f.read()
same = protected_after == protected_before
check("get_dateset.py sha256 unchanged", same,
      hashlib.sha256(protected_after).hexdigest())

n_fail = sum(1 for _, ok in results if not ok)
shutil.rmtree(tmp_dir, ignore_errors=True)   # never leave verification leftovers
print()
print("=" * 72)
print("RESULT: %d checks, %d failed -> %s"
      % (len(results), n_fail, "PASS" if n_fail == 0 else "FAIL"))
print("=" * 72)
sys.exit(1 if n_fail else 0)
