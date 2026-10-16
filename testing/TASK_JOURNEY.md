# FootfallCam Task Decision Journey

## 1. Starting Point

The task is to determine whether a staff member is present in each video frame. The operational definition of staff presence is the visibility of the staff member's name tag.

This means the system is not required to read the name, recognize a person's identity, or classify a person from their appearance. The core signal is the visible `name_tag`.

The assignment also mentions staff location as a bonus output. A detected name-tag bounding box, or its center point, provides a useful location proxy because it identifies the image region where the staff member's tag is visible. A full-person bounding box is stronger localization, but it is not necessary for the primary presence decision.

## 2. Available Data and Initial Constraints

The supplied video is `sample.mp4`, recorded at approximately 25 FPS with 1,341 frames and a resolution of 960 x 720 pixels.

The existing training workflow extracts every fifth frame and uses a one-class YOLO dataset whose class is `name_tag`. This is important for the architecture decision: the available task-specific annotations describe name tags, not complete people.

Two possible processing strategies were considered:

1. **Person-first:** find people first, then search for a name tag inside each detected person region.
2. **Tag-first:** search for name tags directly across the complete frame.

Both approaches are technically reasonable. The decision depends on which upstream errors are most harmful to the assignment's primary objective.

## 3. Person-First Hypothesis

The person-first design would process each frame in several stages:

1. Run a pretrained person detector.
2. Optionally track detected people across frames with ByteTrack.
3. Crop each detected person region.
4. Run the name-tag detector on each crop.
5. Convert the tag coordinates from crop coordinates back to full-frame coordinates.
6. Use the person box or track ID to associate the tag with a particular person.

The intended benefit is that the system could provide a full-person location and potentially accumulate tag evidence per person over time. This would be useful if the main requirement were person identity, person-level tracking, or reliable full-body coordinates.

However, this design introduces a hard dependency: a person must first be detected before the tag detector receives the relevant image region.

## 4. Person-Detection Experiment

Before selecting the final architecture, pretrained YOLO person detection and ByteTrack tracking were tested on `sample.mp4`. The implementation is recorded in `testing/d1.ipynb`, and the configuration history and observations are documented in `testing/person_detector_experiment_log.md`.

The experiment progressed through four configurations:

### Experiment 1: YOLO11n at 640 pixels

The baseline used YOLO11n with an image size of 640, a confidence threshold of 0.25, and ByteTrack. Most detected boxes corresponded to real people, but many seated people were missed. Some nearby people were grouped into one box, and track IDs changed frequently when people were close together or temporarily missed.

### Experiment 2: YOLO11n at 960 pixels

Increasing the image size improved recall during visual inspection and separated at least one previously merged pair. However, seated people near the frame edges were still frequently missed. Some detections included surrounding objects, duplicate boxes appeared in some cases, and tracking remained unstable in crowded areas.

### Experiment 3: YOLO11n at 960 pixels with a lower threshold

The confidence threshold was lowered from 0.25 to 0.15 to investigate whether weaker detections could recover additional seated people. Visual inspection did not show a meaningful recall improvement, while at least one keyboard was incorrectly detected as a person. Track-ID stability also did not improve meaningfully, so this configuration was rejected.

### Experiment 4: YOLO11m at 960 pixels

The larger YOLO11m model produced a slightly stronger observed baseline. Boxes generally captured people more accurately, and track continuity appeared better when several people were close together. Nevertheless, some people were still merged into one box, and the remaining missed detections showed that the upstream person detector was not reliable enough to serve as a required gate for tag detection.

## 5. What the Experiment Established

These experiments were qualitative visual evaluations rather than formal benchmark measurements. They did not establish numerical person-detection precision, recall, tracking accuracy, or tag-detection recall after cropping.

They did establish an important architectural risk for this task: people near the frame edges and some seated people could be missed by the generic person detector. In a person-first pipeline, a missed person means that no crop is created for that person. The name-tag detector therefore gets no opportunity to determine whether the tag is visible.

This can create a false negative before the task-specific detector is even used. The person-first design also adds other possible failure points, including:

- A person box that includes too much surrounding background.
- Two nearby people merged into one box.
- Duplicate boxes for one person.
- A tag cut off by an inaccurate crop boundary.
- Errors when remapping crop coordinates to the original frame.
- Track-ID changes that mix evidence from different people.

These problems do not make person-first unusable. They mean that person detection is not a safe prerequisite when the primary question is simply whether a visible staff tag exists anywhere in the frame.

## 6. Revised Direction: Tag-First

The revised design runs the trained `name_tag` detector directly on the complete video frame:

1. Read the full frame.
2. Detect one or more name-tag regions.
3. Classify the frame as staff-present when at least one accepted tag detection exists.
4. Preserve the detected tag box and confidence as the evidence for that decision.
5. Use the tag box or its center as the available staff-location proxy.

This approach is directly aligned with the annotations and the primary evaluation objective. It does not require a generic person detector to succeed before the task-specific signal can be evaluated. A tag near the frame edge can still be considered as long as the tag detector can see and recognize it.

The decision therefore changed from person-first to tag-first because tag-first reduces the number of upstream conditions required for a positive staff-presence result. It prioritizes detection of the signal the assignment actually asks for over obtaining a full-person box that is not required for the core decision. The implemented optional hybrid mode keeps this ordering: it runs pretrained COCO person localization only after a tag is found, then associates each tag by inclusive tag-center containment. It does not use ByteTrack or identity tracking, and an unmatched tag remains staff-positive.

## 7. Decision Scope

Tag-first is the selected direction for the current task. It is a deliberate choice based on the objective, the available annotations, and the observed weaknesses of the person-first prerequisite.

The tradeoff is explicit: the current direction localizes the visible name tag and optionally reports a matched full-person box, but does not assign the tag to a person identity. Person results are predictions rather than formally validated ground truth. The prior person experiments were qualitative and did not establish formal precision, recall, or tracking metrics.

The person-first experiment remains valuable because it tested a plausible alternative and revealed where that alternative could fail. Recording that experiment makes the final architecture decision reproducible rather than presenting tag-first as an unexplained assumption.

## 8. Source Artifacts

- `IMPLEMENTATION_GUIDE.md`: assignment definition, required outputs, and bonus localization scope.
- `testing/d1.ipynb`: person-detection and tracking experiment implementation.
- `testing/person_detector_experiment_log.md`: configuration history and qualitative observations.
- `dataset/data.yaml`: one-class `name_tag` dataset definition.
- `predict.py`: current full-frame name-tag inference entry point.
