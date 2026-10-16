"""Run the trained name-tag detector over every frame of a video."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
from ultralytics import YOLO


DEFAULT_WEIGHTS = Path("frames_25fps_every_5/runs/name_tag_detection/weights/best.pt")
PERSON_IMAGE_SIZE = 960
PERSON_CLASS_ID = 0


@dataclass
class Detection:
    confidence: float
    box: list[int]
    class_id: int


@dataclass
class FrameResult:
    frame_number: int
    timestamp_seconds: float
    staff_present_raw: bool
    staff_present_smoothed: bool
    confidence: float | None
    name_tag_box_pixels: list[int] | None
    detections: list[dict[str, Any]]
    person_detections: list[dict[str, Any]] = field(default_factory=list)
    tag_associations: list[dict[str, Any]] = field(default_factory=list)
    associated_person_box_pixels: list[int] | None = None
    associated_person_confidence: float | None = None
    association_status: str = "person_stage_disabled"


def read_video_metadata(video_path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Video metadata is invalid: {video_path}")

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps,
    }


def iter_video_frames(video_path: Path) -> Iterable[tuple[int, float, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame_number, frame_number / fps, frame
            frame_number += 1
    finally:
        capture.release()


def load_detector(weights_path: Path) -> YOLO:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    return YOLO(str(weights_path))


def detect_name_tag(
    model: YOLO, frame: Any, confidence_threshold: float
) -> list[Detection]:
    predictions = model(frame, conf=confidence_threshold, verbose=False)
    if not predictions or predictions[0].boxes is None:
        return []

    boxes = predictions[0].boxes
    detections: list[Detection] = []
    for coordinates, confidence, class_id in zip(
        boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
    ):
        detections.append(
            Detection(
                confidence=float(confidence),
                box=[int(round(value)) for value in coordinates],
                class_id=int(class_id),
            )
        )
    return detections


def detect_people(
    model: YOLO, frame: Any, confidence_threshold: float
) -> list[Detection]:
    predictions = model(
        frame,
        classes=[PERSON_CLASS_ID],
        conf=confidence_threshold,
        imgsz=PERSON_IMAGE_SIZE,
        verbose=False,
    )
    if not predictions or predictions[0].boxes is None:
        return []

    boxes = predictions[0].boxes
    detections: list[Detection] = []
    for coordinates, confidence, class_id in zip(
        boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
    ):
        if int(class_id) != PERSON_CLASS_ID:
            continue
        detections.append(
            Detection(
                confidence=float(confidence),
                box=[int(round(value)) for value in coordinates],
                class_id=int(class_id),
            )
        )
    return detections


def box_center(box: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def contains_point(box: list[int], point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = box
    center_x, center_y = point
    return x1 <= center_x <= x2 and y1 <= center_y <= y2


def associate_tag_with_person(
    tag: Detection, people: list[Detection]
) -> Detection | None:
    candidates = [
        person for person in people if contains_point(person.box, box_center(tag.box))
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda person: (
            (person.box[2] - person.box[0]) * (person.box[3] - person.box[1]),
            -person.confidence,
        ),
    )


def build_tag_associations(
    tags: list[Detection], people: list[Detection], person_stage_enabled: bool
) -> list[dict[str, Any]]:
    associations: list[dict[str, Any]] = []
    for tag in tags:
        person = associate_tag_with_person(tag, people) if person_stage_enabled else None
        associations.append(
            {
                "tag_box_pixels": tag.box,
                "person_box_pixels": person.box if person else None,
                "person_confidence": person.confidence if person else None,
                "status": "matched" if person else ("no_person_match" if person_stage_enabled else "person_stage_disabled"),
            }
        )
    return associations


def classify_staff_presence(detections: list[Detection]) -> bool:
    return bool(detections)


def smooth_temporal_predictions(
    results: list[FrameResult], max_gap_frames: int
) -> None:
    """Fill short raw-negative gaps between positive detections in place."""
    if max_gap_frames <= 0:
        for result in results:
            result.staff_present_smoothed = result.staff_present_raw
        return

    for result in results:
        result.staff_present_smoothed = result.staff_present_raw

    index = 0
    while index < len(results):
        if results[index].staff_present_raw:
            index += 1
            continue

        gap_start = index
        while index < len(results) and not results[index].staff_present_raw:
            index += 1
        gap_end = index
        gap_length = gap_end - gap_start
        before_positive = gap_start > 0 and results[gap_start - 1].staff_present_raw
        after_positive = gap_end < len(results) and results[gap_end].staff_present_raw

        if before_positive and after_positive and gap_length <= max_gap_frames:
            for gap_index in range(gap_start, gap_end):
                results[gap_index].staff_present_smoothed = True


def render_visualization(
    frame: Any,
    result: FrameResult,
    fps: float,
) -> Any:
    output = frame.copy()
    associated_people: dict[tuple[int, int, int, int], float] = {}
    for association in result.tag_associations:
        person_box = association["person_box_pixels"]
        person_confidence = association["person_confidence"]
        if person_box is not None and person_confidence is not None:
            associated_people[tuple(person_box)] = person_confidence

    for person_box, person_confidence in associated_people.items():
        x1, y1, x2, y2 = person_box
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 140, 0), 2)
        cv2.putText(
            output,
            f"staff {person_confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 140, 0),
            2,
            cv2.LINE_AA,
        )

    for detection in result.detections:
        x1, y1, x2, y2 = detection["box"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 210, 40), 2)
        label = f"name_tag {detection['confidence']:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 210, 40),
            2,
            cv2.LINE_AA,
        )

    status = "STAFF PRESENT" if result.staff_present_raw else "NO STAFF TAG"
    cv2.putText(
        output,
        f"Frame {result.frame_number}  {result.timestamp_seconds:.2f}s  {status}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 220, 255) if result.staff_present_raw else (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return output


def write_results(
    results: list[FrameResult], metadata: dict[str, Any], output_dir: Path
) -> None:
    json_payload = {
        "metadata": metadata,
        "predictions": [asdict(result) for result in results],
    }
    (output_dir / "predictions.json").write_text(
        json.dumps(json_payload, indent=2), encoding="utf-8"
    )

    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        fieldnames = [
            "frame_number",
            "timestamp_seconds",
            "staff_present_raw",
            "staff_present_smoothed",
            "confidence",
            "name_tag_box_pixels",
            "detections",
            "person_detections",
            "tag_associations",
            "associated_person_box_pixels",
            "associated_person_confidence",
            "association_status",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["name_tag_box_pixels"] = json.dumps(row["name_tag_box_pixels"])
            row["detections"] = json.dumps(row["detections"])
            row["person_detections"] = json.dumps(row["person_detections"])
            row["tag_associations"] = json.dumps(row["tag_associations"])
            row["associated_person_box_pixels"] = json.dumps(
                row["associated_person_box_pixels"]
            )
            writer.writerow(row)


def save_examples(
    examples: dict[str, Any], output_dir: Path, results: list[FrameResult]
) -> None:
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    for name, frame in examples.items():
        cv2.imwrite(str(examples_dir / f"{name}.jpg"), frame)

    positives = [result.frame_number for result in results if result.staff_present_raw]
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "raw_positive_frame_count": len(positives),
                "raw_positive_frame_first": positives[0] if positives else None,
                "raw_positive_frame_last": positives[-1] if positives else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_inference(
    video_path: Path,
    weights_path: Path,
    output_dir: Path,
    confidence_threshold: float,
    max_gap_frames: int,
    save_video: bool,
    person_weights_path: Path | None = None,
    person_confidence_threshold: float = 0.25,
) -> None:
    video_path = video_path.resolve()
    weights_path = weights_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_video_metadata(video_path)
    model = load_detector(weights_path)
    person_model = (
        load_detector(person_weights_path.resolve())
        if person_weights_path is not None
        else None
    )
    metadata = {
        **metadata,
        "person_localization": {
            "enabled": person_model is not None,
            "weights": str(person_weights_path.resolve())
            if person_weights_path is not None
            else None,
            "confidence_threshold": person_confidence_threshold,
            "image_size": PERSON_IMAGE_SIZE,
            "class_id": PERSON_CLASS_ID,
            "tracking": False,
        },
    }
    fps = float(metadata["fps"])
    writer = None
    examples: dict[str, Any] = {}
    results: list[FrameResult] = []

    if save_video:
        video_output = output_dir / "annotated_video.mp4"
        writer = cv2.VideoWriter(
            str(video_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (int(metadata["width"]), int(metadata["height"])),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create annotated video: {video_output}")

    try:
        for frame_number, timestamp, frame in iter_video_frames(video_path):
            detections = detect_name_tag(model, frame, confidence_threshold)
            person_detections = (
                detect_people(person_model, frame, person_confidence_threshold)
                if person_model is not None and detections
                else []
            )
            associations = build_tag_associations(
                detections, person_detections, person_model is not None
            )
            detection_dicts = [asdict(detection) for detection in detections]
            person_detection_dicts = [
                asdict(detection) for detection in person_detections
            ]
            best_detection = detections[0] if detections else None
            best_association = associations[0] if associations else None
            result = FrameResult(
                frame_number=frame_number,
                timestamp_seconds=timestamp,
                staff_present_raw=classify_staff_presence(detections),
                staff_present_smoothed=False,
                confidence=best_detection.confidence if best_detection else None,
                name_tag_box_pixels=best_detection.box if best_detection else None,
                detections=detection_dicts,
                person_detections=person_detection_dicts,
                tag_associations=associations,
                associated_person_box_pixels=(
                    best_association["person_box_pixels"]
                    if best_association
                    else None
                ),
                associated_person_confidence=(
                    best_association["person_confidence"]
                    if best_association
                    else None
                ),
                association_status=(
                    best_association["status"]
                    if best_association
                    else "person_stage_disabled"
                ),
            )
            results.append(result)
            annotated = render_visualization(frame, result, fps)
            if result.staff_present_raw and "positive" not in examples:
                examples["positive"] = annotated
            if not result.staff_present_raw and "negative" not in examples:
                examples["negative"] = annotated
            if writer is not None:
                writer.write(annotated)
    finally:
        if writer is not None:
            writer.release()

    smooth_temporal_predictions(results, max_gap_frames)
    write_results(results, metadata, output_dir)
    save_examples(examples, output_dir, results)
    print(f"Processed {len(results)} frames")
    print(f"Metadata: {json.dumps(metadata)}")
    print(f"Outputs: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--person-weights",
        type=Path,
        default=None,
        help="Optional YOLO weights for conditional COCO person localization.",
    )
    parser.add_argument("--person-confidence", type=float, default=0.25)
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=0,
        help="Fill at most this many raw-negative frames between positives; 0 disables smoothing.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated_video.mp4 while still writing predictions and examples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.confidence <= 1:
        raise SystemExit("--confidence must be between 0 and 1")
    if not 0 <= args.person_confidence <= 1:
        raise SystemExit("--person-confidence must be between 0 and 1")
    if args.max_gap_frames < 0:
        raise SystemExit("--max-gap-frames cannot be negative")
    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")
    run_inference(
        video_path=args.video,
        weights_path=args.weights,
        output_dir=args.output,
        confidence_threshold=args.confidence,
        max_gap_frames=args.max_gap_frames,
        save_video=not args.no_video,
        person_weights_path=args.person_weights,
        person_confidence_threshold=args.person_confidence,
    )


if __name__ == "__main__":
    main()