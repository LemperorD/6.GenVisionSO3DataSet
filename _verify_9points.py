# -*- coding: utf-8 -*-
"""
Offline self-check for the 9-point geometry (ASCII-only source on purpose).
Run:  D:\\PX4PSP\\Python38\\python.exe _verify_9points.py

Checks:
  1. getUav9Point() output order matches POINT_OFFSETS in get_dateset.py
  2. envelope box size
  3. pixel projection of the 9 points
  4. draw_9_points() renders correctly -> _verify_9points.png
No UE4 / RflySim3D connection is needed. No dataset folder is created.
"""
import math
import os
import sys
import types

import numpy as np
import cv2

# ---- stub out the RflySim API modules so nothing tries to reach UE4 ----
fake_vis = types.ModuleType("VisionCaptureApi")
fake_vis.VisionCaptureApi = lambda *a, **k: types.SimpleNamespace()
fake_ue = types.ModuleType("UE4CtrlAPI")
fake_ue.UE4CtrlAPI = lambda *a, **k: types.SimpleNamespace()
sys.modules["VisionCaptureApi"] = fake_vis
sys.modules["UE4CtrlAPI"] = fake_ue

TARGET = "test_dateset.py"   # 只读取测试副本；get_dateset.py 是原始版本，禁止读写
PROTECTED = "get_dateset.py"

# 只在内存中读取，且读到内存后立即关闭句柄（不写、不改 mtime）
with open(TARGET, encoding="utf-8") as f:
    src = f.read()

# 红线保护：get_dateset.py 必须保持原版
with open(PROTECTED, "rb") as f:
    _protected_snapshot = f.read()
print("[guard] %s read-only baseline: %d bytes, sha256=%s"
      % (PROTECTED, len(_protected_snapshot),
         __import__("hashlib").sha256(_protected_snapshot).hexdigest()))

# ---- take everything up to the main loop (functions + constants only) ----
cut = src.index("startTime = time.time()")
head = src[:cut]

lines = []


class _GuardedOs(types.ModuleType):
    """Forbid real directory creation inside the probe."""

    @staticmethod
    def makedirs(*a, **k):
        return None


_real_os = os
guard_os = _GuardedOs("os")
guard_os.__dict__.update(
    {k: v for k, v in _real_os.__dict__.items() if k != "makedirs"}
)

for ln in head.splitlines():
    s = ln.strip()
    if s.startswith("import os") or s.startswith("from os"):
        continue
    if s.startswith("ue.sendUE4Cmd") or s.startswith("ue.waitForMapLoaded") or \
       s.startswith("ue.sendUE4Pos") or s.startswith("vis.") or \
       s.startswith("isSuss ="):
        continue
    lines.append(ln)

ns = {"__name__": "_geo_probe", "isSuss": True, "os": guard_os}
exec("\n".join(lines), ns)

eul2rot = ns["eul2rot"]
getUav9Point = ns["getUav9Point"]
project_point = ns["project_point"]
draw_9_points = ns["draw_9_points"]
POINT_OFFSETS = ns["POINT_OFFSETS"]
center_idx = ns["POINT_CENTER_INDEX"]
copterCenterHeight = ns["copterCenterHeight"]
px, py, focal = ns["px"], ns["py"], ns["focal"]
pic_w, pic_h = ns["pic_w"], ns["pic_h"]

# ---- reproduce one typical main-loop call ----
PosInit = [0, 0, -8.086 - 4]
angCopterE = [0, 0, 0]
cameraPosForUav = [0.03, 0, 0]
TargePos = [2.03, 0.3, -12.086 + 0.2]
TargeAng = [0.2, -0.15, math.pi / 2 + 0.3]

camere_pos = PosInit + np.transpose(np.dot(np.linalg.inv(
    eul2rot(angCopterE)), np.transpose(cameraPosForUav)))

points = getUav9Point(TargePos, copterCenterHeight, TargeAng)

print("=" * 72)
print("[1] point-order check: getUav9Point() vs POINT_OFFSETS")
print("=" * 72)
center_world = np.array(TargePos) + [0, 0, -copterCenterHeight]
all_ok = True
for i, (pt, off) in enumerate(zip(points, POINT_OFFSETS)):
    # getUav9Point uses  center + inv(R) @ offset, so invert with R itself
    delta = np.dot(eul2rot(TargeAng), (np.array(pt) - center_world))
    err = float(np.max(np.abs(delta - np.array(off))))
    if err >= 1e-9:
        all_ok = False
    tag = "C" if i == center_idx else str(i)
    print("  idx %-2s %-4s actual(%+.3f,%+.3f,%+.3f)  declared(%+.3f,%+.3f,%+.3f)"
          % (tag, "OK" if err < 1e-9 else "BAD",
             delta[0], delta[1], delta[2], off[0], off[1], off[2]))
print("  -> order consistency: %s" % ("PASS" if all_ok else "FAIL"))

ext = [max(o[k] for o in POINT_OFFSETS) - min(o[k] for o in POINT_OFFSETS)
       for k in range(3)]
print("  envelope box: X=%.3f m  Y=%.3f m  Z=%.3f m" % tuple(ext))

print()
print("=" * 72)
print("[2] projection check (target pos = %s)" % (TargePos,))
print("=" * 72)
pic_sit = np.array([project_point(p, camere_pos, angCopterE) for p in points])
names = ["0(lowest)", "1(front-right)", "2(rear-right)", "3(front-left)",
         "4(rear-left)", "5(top-fr)", "6(top-rr)", "7(top-fl)", "C(center)"]
n_out = 0
for k in range(9):
    cx, cy = pic_sit[k]
    inside = (0 <= cx < pic_w) and (0 <= cy < pic_h)
    n_out += 0 if inside else 1
    print("  %-14s -> (%5d, %5d)  %s"
          % (names[k], cx, cy, "inside" if inside else "OUTSIDE *"))
x1y1 = [min(pic_sit[:, 0]), min(pic_sit[:, 1])]
x2y2 = [max(pic_sit[:, 0]), max(pic_sit[:, 1])]
print("  bbox (before clipping) = [%d, %d, %d, %d]" % (x1y1[0], x1y1[1], x2y2[0], x2y2[1]))
print("  points outside frame: %d" % n_out)

print()
print("=" * 72)
print("[3] draw_9_points() render check -> _verify_9points.png")
print("=" * 72)
canvas = np.full((pic_h, pic_w, 3), 60, np.uint8)
canvas = draw_9_points(canvas, pic_sit, detail=True)
cv2.rectangle(canvas, (int(x1y1[0]), int(x1y1[1])),
              (int(x2y2[0]), int(x2y2[1])), (0, 0, 255), 1)
cv2.imwrite("_verify_9points.png", canvas)
non_bg = int(np.count_nonzero(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) != 60))
print("  wrote _verify_9points.png (%dx%d), non-background pixels = %d"
      % (canvas.shape[1], canvas.shape[0], non_bg))

# ---- 红线校验：确认 get_dateset.py 全程未被触碰 ----
with open(PROTECTED, "rb") as f:
    after = f.read()
same = (after == _protected_snapshot)
print()
print("[guard] %s unchanged during this run: %s" % (PROTECTED, same))
print("  RESULT: %s" % ("PASS" if (all_ok and non_bg > 0 and same) else "FAIL"))
