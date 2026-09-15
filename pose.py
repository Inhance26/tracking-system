"""COCO-17 skeleton schema, quality scoring, and the usability gate.

The pose model gives 17 keypoints per person. On this footage that is not the
same as 17 *useful* keypoints: the people at the back of the shop are ~12x40
px (see the note in app.py), and a skeleton fitted to a 40 px-tall blob is
noise wearing the shape of a person. Feed those to an action-recognition model
and you are training it on invented motion.

So every pose carries a quality score, and a gate decides whether it is worth
keeping. Nothing here throws a pose away - pipeline.py still draws whatever the
model produced - but `usable` is what a downstream ST-GCN window should filter
on, and pose_audit.py exists to help you pick the thresholds from your own
footage rather than from these defaults.

Keypoints are stored NORMALISED (0-1 of frame width/height), matching the
convention zones.py and detector.py already use, so they survive a resize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# COCO-17 order, exactly as the Ultralytics pose models emit it. Downstream
# ST-GCN/PoseC3D pretrained weights assume this order - do not reorder.
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
NUM_KEYPOINTS = len(KEYPOINT_NAMES)

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

# The head keypoints. reid.py will mask this region out of the crop before
# computing an appearance embedding, which is what makes "no face recognition"
# a property of the code rather than a promise.
HEAD_KEYPOINTS = (0, 1, 2, 3, 4)

# Torso + limbs, ignoring the face. These are the joints an action classifier
# actually leans on, and the ones big enough to survive a small bounding box,
# so quality is judged on these rather than on whether an ear was found.
CORE_KEYPOINTS = tuple(range(5, NUM_KEYPOINTS))

# Bones, for drawing. Pairs of keypoint indices.
SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),              # face
    (5, 6),                                       # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),              # arms
    (5, 11), (6, 12), (11, 12),                   # torso
    (11, 13), (13, 15), (12, 14), (14, 16),       # legs
)

# --- gate defaults ---------------------------------------------------------
# Deliberately conservative starting points, NOT measured truths. Run
# pose_audit.py against your own footage and move them.
#
# MIN_HEIGHT_PX 80: below roughly this, adjacent joints land on the same pixel
# and elbow/wrist positions are guesses. The bundled clip's distant workers
# (~40 px) sit well under it, so expect them to be rejected - that is the gate
# working, not failing.
MIN_HEIGHT_PX = 80.0
MIN_KP_CONF = 0.5        # per-keypoint confidence to count as "seen"
MIN_CORE_VISIBLE = 8     # of the 12 core joints


@dataclass
class PoseQuality:
    """Why a pose was or wasn't accepted. Cheap to build, safe to serialise."""

    height_px: float          # bounding-box height in SOURCE pixels
    visible_core: int         # core joints above MIN_KP_CONF
    mean_core_conf: float     # mean confidence over the core joints
    usable: bool

    @property
    def score(self) -> float:
        """0-1 summary, for ranking rather than for gating.

        Height is capped at twice the threshold: a person filling the frame is
        not four times more useful than one comfortably over the line.
        """
        h = min(1.0, self.height_px / (2.0 * MIN_HEIGHT_PX))
        v = self.visible_core / max(1, len(CORE_KEYPOINTS))
        return round(h * 0.4 + v * 0.4 + self.mean_core_conf * 0.2, 3)

    def as_dict(self) -> dict:
        return {
            "height_px": round(self.height_px, 1),
            "visible_core": self.visible_core,
            "mean_core_conf": round(self.mean_core_conf, 3),
            "usable": self.usable,
            "score": self.score,
        }


def assess(bbox: Sequence[float], kp_conf: Optional[Sequence[float]],
           frame_h: int,
           min_height_px: float = MIN_HEIGHT_PX,
           min_kp_conf: float = MIN_KP_CONF,
           min_core_visible: int = MIN_CORE_VISIBLE) -> PoseQuality:
    """Score one person's pose. `bbox` is normalised (x1, y1, x2, y2)."""
    height_px = max(0.0, (bbox[3] - bbox[1])) * float(frame_h)
    if not kp_conf:
        return PoseQuality(height_px, 0, 0.0, False)

    core = [float(kp_conf[i]) for i in CORE_KEYPOINTS if i < len(kp_conf)]
    visible = sum(1 for c in core if c >= min_kp_conf)
    mean_conf = sum(core) / len(core) if core else 0.0
    usable = height_px >= min_height_px and visible >= min_core_visible
    return PoseQuality(height_px, visible, mean_conf, usable)


def normalise(keypoints: Sequence[Sequence[float]],
              kp_conf: Optional[Sequence[float]] = None
              ) -> Optional[List[List[float]]]:
    """Root-centre on the mid-hip and scale by torso length.

    Removes where the person stands and how big they appear, leaving only the
    shape of the pose - which is what an action classifier should be learning
    from. Not used by phase 1's pipeline; sequences.py will call it when
    building ST-GCN windows, and it lives here so the definition of "normalised
    pose" has exactly one home.

    Returns None when the hips or shoulders are missing, since there is then no
    trustworthy origin or scale to divide by.
    """
    if not keypoints or len(keypoints) < NUM_KEYPOINTS:
        return None

    def seen(i: int) -> bool:
        return kp_conf is None or (i < len(kp_conf) and kp_conf[i] >= MIN_KP_CONF)

    if not (seen(LEFT_HIP) and seen(RIGHT_HIP)
            and seen(LEFT_SHOULDER) and seen(RIGHT_SHOULDER)):
        return None

    hip = ((keypoints[LEFT_HIP][0] + keypoints[RIGHT_HIP][0]) / 2.0,
           (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2.0)
    shoulder = ((keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0]) / 2.0,
                (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1]) / 2.0)

    torso = ((shoulder[0] - hip[0]) ** 2 + (shoulder[1] - hip[1]) ** 2) ** 0.5
    if torso < 1e-6:            # degenerate - a top-down view, or bad joints
        return None

    return [[(x - hip[0]) / torso, (y - hip[1]) / torso] for x, y in keypoints]


def to_pixels(keypoints: Sequence[Sequence[float]], width: int, height: int
              ) -> List[Tuple[int, int]]:
    """Normalised keypoints -> pixel coords, mirroring Zone.to_pixels()."""
    return [(int(round(x * width)), int(round(y * height))) for x, y in keypoints]
