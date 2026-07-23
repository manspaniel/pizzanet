#!/usr/bin/env python3
"""Compare a vio-replay trace against ARKit ground-truth poses.

Usage:
    ./scripts/compare-arkit-ground-truth.py datasets/ar-recordings/<id> [trace.ndjson]

Runs vio-replay on the recording (unless an existing trace is given), pairs
frames by frameId with the recording's arkit-poses.ndjson, then aligns the two
trajectories with the transform that is genuinely unobservable between them —
yaw rotation, translation, and (reported both ways) scale — and prints:

  - scale ratio (ours / ARKit): the metric-scale error. 1.0 = perfect scale.
  - ATE RMSE after similarity alignment: shape error independent of scale.
  - ATE RMSE after rigid alignment (scale forced to 1): total metric error.
  - endpoint error and max ground-truth displacement (the "tape measure").

Both trajectories are gravity-aligned y-up, so only yaw can differ in rotation.
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_ndjson(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_replay(recording: Path) -> Path:
    trace = Path(tempfile.mkstemp(suffix=".ndjson")[1])
    subprocess.run(
        [
            str(REPO / "target/release/vio-replay"),
            "recording",
            str(recording),
            "--frames",
            str(recording / "tracker-luma.gray"),
            "--output",
            "/dev/null",
            "--trace-output",
            str(trace),
        ],
        check=True,
        capture_output=True,
    )
    return trace


def align(ours: list[list[float]], truth: list[list[float]], fit_scale: bool):
    """Fit s·R_y(theta)·p + t ≈ q by least squares; return (s, theta, rmse)."""
    n = len(ours)
    mean_p = [sum(p[i] for p in ours) / n for i in range(3)]
    mean_q = [sum(q[i] for q in truth) / n for i in range(3)]
    cp = [[p[i] - mean_p[i] for i in range(3)] for p in ours]
    cq = [[q[i] - mean_q[i] for i in range(3)] for q in truth]

    # Yaw via the xz-plane cross-correlation (treat xz as complex numbers).
    real = sum(p[0] * q[0] + p[2] * q[2] for p, q in zip(cp, cq))
    imag = sum(p[2] * q[0] - p[0] * q[2] for p, q in zip(cp, cq))
    theta = math.atan2(imag, real)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def rotate(p):
        return [cos_t * p[0] + sin_t * p[2], p[1], -sin_t * p[0] + cos_t * p[2]]

    rotated = [rotate(p) for p in cp]
    if fit_scale:
        num = sum(r[i] * q[i] for r, q in zip(rotated, cq) for i in range(3))
        den = sum(r[i] * r[i] for r in rotated for i in range(3))
        scale = num / den if den > 1e-12 else 1.0
    else:
        scale = 1.0
    squared = sum(
        (scale * r[i] - q[i]) ** 2 for r, q in zip(rotated, cq) for i in range(3)
    )
    rmse = math.sqrt(squared / n)
    return scale, math.degrees(theta), rmse


def main() -> None:
    recording = Path(sys.argv[1])
    poses_path = recording / "arkit-poses.ndjson"
    if not poses_path.exists():
        sys.exit(f"{poses_path} not found — is this a native ARKit recording?")
    trace_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else run_replay(recording)
    )

    truth_by_frame = {p["frameId"]: p["position"] for p in load_ndjson(poses_path)}
    pairs = [
        (t["position"], truth_by_frame[t["frameId"]])
        for t in load_ndjson(trace_path)
        if t["frameId"] in truth_by_frame
    ]
    if len(pairs) < 30:
        sys.exit(f"only {len(pairs)} paired frames — not enough to compare")
    ours = [p for p, _ in pairs]
    truth = [q for _, q in pairs]

    scale, yaw, sim_rmse = align(ours, truth, fit_scale=True)
    _, _, rigid_rmse = align(ours, truth, fit_scale=False)

    def dist(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    truth_start = truth[0]
    max_displacement = max(dist(q, truth_start) for q in truth)
    endpoint_truth = dist(truth[-1], truth_start)
    endpoint_ours = dist(ours[-1], ours[0])

    print(f"paired frames           {len(pairs)}")
    print(f"ARKit max displacement  {max_displacement:.3f} m   (the tape measure)")
    print(f"ARKit endpoint offset   {endpoint_truth:.3f} m")
    print(f"our endpoint offset     {endpoint_ours:.3f} m")
    print(f"scale ratio (ARKit/ours after yaw fit): {scale:.3f}   (1.0 = metric)")
    print(f"yaw alignment           {yaw:+.1f} deg")
    print(f"ATE RMSE (similarity)   {sim_rmse:.3f} m   (shape error, scale-free)")
    print(f"ATE RMSE (rigid)        {rigid_rmse:.3f} m   (total metric error)")


if __name__ == "__main__":
    main()
