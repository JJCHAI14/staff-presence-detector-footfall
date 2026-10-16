# FootfallCam AI Evaluation — Person Detection & Tracking Experiment Log

## 1. Objective

Evaluate pretrained YOLO person detection and ByteTrack tracking on `sample.mp4` before implementing the downstream name-tag detection pipeline.

**Pipeline:** Video frames → YOLO person detector → ByteTrack tracker → Bounding boxes with track IDs

**Video characteristics**
- Resolution: 960 × 720
- Frame rate: 25 FPS
- Total frames: 1,341 (~53.6 seconds)
- Hardware: CPU-only, 32 GB RAM
- Frameworks: Ultralytics YOLO, ByteTrack, OpenCV

**Initial evaluation objectives**
1. Detect seated and standing people reliably from an overhead 3D-sensor view.
2. Produce reasonably accurate bounding boxes around individual people.
3. Avoid false-positive person detections.
4. Maintain consistent track IDs as people move, overlap, or remain seated.

---

## 2. Experiment history

### Experiment 1 — Baseline pretrained model

**Configuration**
- Model: `yolo11n.pt`
- Image size: 640
- Confidence threshold: 0.25
- Tracker: ByteTrack (`bytetrack.yaml`, default settings)
- Detection class: Person (COCO class 0)

**Observations**
- Most detected boxes corresponded to actual people.
- Few obvious false-positive detections were observed.
- Many seated people were missed unless they made large gestures.
- Two people sitting at opposite ends of a table could be grouped into one bounding box.
- Track IDs frequently changed when people stayed close together or were temporarily missed and detected again.

**Interpretation**

The baseline model provided a useful starting point but struggled with stationary people and crowded overhead-view scenes. Missed detections and merged bounding boxes also limited the tracker's ability to maintain reliable identities.

**Decision:** Increase the inference image size to investigate whether retaining more image detail improves detection.

---

### Experiment 2 — Increase image size

**Configuration**
- Model: `yolo11n.pt`
- Image size: 960
- Confidence threshold: 0.25
- Tracker: ByteTrack (`bytetrack.yaml`, default settings)

**Observations**
- Recall increased compared with Experiment 1.
- Many boxes still corresponded to actual people, with no obvious widespread false-positive problem.
- Some bounding boxes included surrounding objects, such as monitors.
- In a few cases, one person received two simultaneous boxes: one covering the person and another covering the person together with a nearby object.
- The previously observed pair of people grouped into one box was separated successfully, but a different nearby pair exhibited the same problem.
- Many seated people, particularly near the video edges, remained undetected.
- Track IDs still changed or reset, especially when several people stayed close together.

**Interpretation**

Increasing the image size appeared to improve recall and helped separate people in at least one previously problematic example. However, the improvement was incomplete and did not eliminate duplicate detections, merged people, or tracking failures.

**Decision:** Lower the confidence threshold to investigate whether additional low-confidence detections could recover more seated people.

---

### Experiment 3 — Lower confidence threshold

**Configuration**
- Model: `yolo11n.pt`
- Image size: 960
- Confidence threshold: 0.15
- Tracker: ByteTrack (`bytetrack.yaml`, default settings)

**Observations**
- Recall did not improve by a meaningful amount according to visual inspection.
- A keyboard was incorrectly detected as a person.
- No meaningful improvement in track-ID stability was observed.

**Interpretation**

The lower threshold admitted at least one additional false-positive detection without recovering enough additional people to justify the trade-off.

**Decision:** Reject this configuration and restore the confidence threshold to 0.25.

---

### Experiment 4 — Increase model capacity

**Configuration**
- Model: `yolo11m.pt`
- Image size: 960
- Confidence threshold: 0.25
- Tracker: ByteTrack (`bytetrack.yaml`, default settings)

**Observations**
- Recall increased slightly compared with the previous `yolo11n.pt` configuration at image size 960 and confidence threshold 0.25.
- Bounding boxes generally captured people relatively accurately.
- A few examples still showed two people grouped into the same bounding box.
- Track-ID continuity improved: the tracker maintained identities more reliably, even when four or five people stayed close together.

**Interpretation**

The larger model appears to provide a better detection-and-tracking baseline for this clip. However, the remaining merged boxes indicate that the model still struggles to separate some nearby people. Improved ID continuity is an observed outcome of the combined detector-and-tracker pipeline; it does not by itself prove that the detector alone improved identity tracking.

**Decision:** Retain this configuration as the current preferred candidate for further evaluation.

---

## 3. Configuration comparison

| Experiment | Model | Image size | Confidence | Main observation | Decision |
|---|---|---:|---:|---|---|
| 1 | YOLO11n | 640 | 0.25 | Many seated people missed; frequent ID changes | Baseline |
| 2 | YOLO11n | 960 | 0.25 | Recall improved; some remaining merged/duplicate boxes | Retain higher resolution |
| 3 | YOLO11n | 960 | 0.15 | Negligible recall gain; keyboard false positive | Reject |
| 4 | YOLO11m | 960 | 0.25 | Slightly higher recall; better boxes and ID continuity | Preferred candidate |

*These results are based on qualitative visual observations. No numerical precision, recall, or tracking accuracy scores have been established yet.*

---

## 4. Current preferred configuration

```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

results = model.track(
    frame,
    persist=True,
    tracker="bytetrack.yaml",
    classes=[0],
    conf=0.25,
    imgsz=960,
    verbose=False,
)
```

This configuration is preferred because it provides a better observed balance between person detection, bounding-box quality, and ID continuity.

The configuration has not yet been formally validated against manually annotated ground truth.

---

## 5. Remaining limitations

1. **Missed detections:** Some seated people, especially near the frame boundaries, remain undetected.
2. **Merged detections:** Two nearby people can occasionally be represented by one bounding box.
3. **Duplicate detections:** Some people previously received two boxes with different spatial coverage.
4. **ID switches and fragmentation:** Tracking is improved but has not been proven free of identity switches or ID resets.
5. **Name-tag visibility:** Person detection does not guarantee that a person's name tag is visible or that the eventual tag detector can localize it.

These issues matter because the downstream name-tag detector will operate on person detections, and track-level voting will aggregate its predictions across frames. Missed detections reduce available evidence, while merged boxes and ID switches can mix evidence from different people.
