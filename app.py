from __future__ import annotations

import argparse
import math
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".jp2"}
DEFAULT_DPI = 300
DEFAULT_FACE_HEIGHT_RATIO = 0.50
DEFAULT_FALLBACK_FACE_WIDTH_RATIO = 0.34
DEFAULT_FALLBACK_FACE_HEIGHT_RATIO = 0.40
DEFAULT_FALLBACK_FACE_TOP_RATIO = 0.16
DEFAULT_HEAD_TOP_EXPANSION = 0.52
DEFAULT_HEAD_BOTTOM_EXPANSION = 0.34
DEFAULT_HEAD_SIDE_EXPANSION = 0.20
DEFAULT_HEAD_HEIGHT_RATIO = 0.64
DEFAULT_HEAD_ANCHOR_Y = 0.44
FACE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


@dataclass
class OutputSpec:
    width_px: int
    height_px: int
    units: str
    dpi: int
    face_height_ratio: float


FACE_CASCADE = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))


def parse_positive_int(raw: str, label: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be a whole number.") from exc
    if value <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    return value


def parse_positive_float(raw: str, label: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if value <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    return value


def prompt_text(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{label}{suffix}: ").strip()
    if raw:
        return raw
    if default is not None:
        return default
    raise ValueError(f"{label} is required.")


def prompt_spec() -> tuple[Path, Path, OutputSpec]:
    input_dir = Path(prompt_text("Input folder")).expanduser().resolve()
    output_dir = Path(prompt_text("Output folder", str(input_dir / "processed_profiles"))).expanduser().resolve()
    units = prompt_text("Units (pixels/inches)", "pixels").strip().lower()
    width_raw = prompt_text("Width", "600")
    height_raw = prompt_text("Height", "600")
    dpi_raw = prompt_text("DPI", str(DEFAULT_DPI))
    face_fill_raw = prompt_text("Face fill 0.10-0.90", str(DEFAULT_FACE_HEIGHT_RATIO))

    spec = build_output_spec(width_raw, height_raw, units, dpi_raw, face_fill_raw)
    return input_dir, output_dir, spec


def build_output_spec(width_raw: str, height_raw: str, units: str, dpi_raw: str, face_fill_raw: str) -> OutputSpec:
    face_height_ratio = parse_positive_float(face_fill_raw, "Face fill")
    if face_height_ratio <= 0.1 or face_height_ratio >= 0.9:
        raise ValueError("Face fill must be between 0.10 and 0.90.")

    dpi = parse_positive_int(dpi_raw, "DPI")
    units = units.strip().lower()

    if units == "inches":
        width_in = parse_positive_float(width_raw, "Width")
        height_in = parse_positive_float(height_raw, "Height")
        width_px = max(1, round(width_in * dpi))
        height_px = max(1, round(height_in * dpi))
        return OutputSpec(width_px, height_px, units, dpi, face_height_ratio)

    if units != "pixels":
        raise ValueError("Units must be either 'pixels' or 'inches'.")

    width_px = parse_positive_int(width_raw, "Width")
    height_px = parse_positive_int(height_raw, "Height")
    return OutputSpec(width_px, height_px, units, dpi, face_height_ratio)


def parse_args() -> tuple[Path, Path, OutputSpec]:
    parser = argparse.ArgumentParser(
        description="Batch process profile photos with face-aware centering on macOS using Python 3.14."
    )
    parser.add_argument("--input-dir", type=str, help="Folder containing source photos.")
    parser.add_argument("--output-dir", type=str, help="Folder for processed photos and the zip archive.")
    parser.add_argument("--width", type=str, help="Output width in pixels or inches.")
    parser.add_argument("--height", type=str, help="Output height in pixels or inches.")
    parser.add_argument("--units", type=str, default="pixels", choices=["pixels", "inches"], help="Size units.")
    parser.add_argument("--dpi", type=str, default=str(DEFAULT_DPI), help="DPI for inch conversion and metadata.")
    parser.add_argument(
        "--face-fill",
        type=str,
        default=str(DEFAULT_FACE_HEIGHT_RATIO),
        help="How much of the final crop height the face should occupy.",
    )
    args = parser.parse_args()

    if not args.input_dir or not args.output_dir or not args.width or not args.height:
        return prompt_spec()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    spec = build_output_spec(args.width, args.height, args.units, args.dpi, args.face_fill)
    return input_dir, output_dir, spec


def list_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS)


def detect_primary_face(image_path: Path) -> dict | None:
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to decode image for face detection: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_face = max(40, min(gray.shape[:2]) // 8)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(min_face, min_face),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    return {"x": float(x), "y": float(y), "width": float(w), "height": float(h)}


def read_image_size(image_path: Path) -> tuple[int, int]:
    image = load_image(image_path)
    height, width = image.shape[:2]
    return width, height


def fit_crop_within(image_width: int, image_height: int, aspect_ratio: float) -> tuple[float, float]:
    candidate_width = image_height * aspect_ratio
    candidate_height = image_height
    if candidate_width <= image_width:
        return candidate_width, candidate_height
    return image_width, image_width / aspect_ratio


def clamp_crop(
    center_x: float,
    center_y: float,
    crop_width: float,
    crop_height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    left = center_x - crop_width / 2
    top = center_y - crop_height / 2
    left = max(0.0, min(left, image_width - crop_width))
    top = max(0.0, min(top, image_height - crop_height))
    return (
        int(round(left)),
        int(round(top)),
        int(round(left + crop_width)),
        int(round(top + crop_height)),
    )


def estimate_fallback_face_box(image_width: int, image_height: int) -> dict:
    portrait_scale = image_height / max(image_width, 1)
    width_ratio = DEFAULT_FALLBACK_FACE_WIDTH_RATIO
    height_ratio = DEFAULT_FALLBACK_FACE_HEIGHT_RATIO
    top_ratio = DEFAULT_FALLBACK_FACE_TOP_RATIO

    if portrait_scale >= 1.35:
        width_ratio = 0.36
        height_ratio = 0.42
        top_ratio = 0.14
    elif portrait_scale <= 1.05:
        width_ratio = 0.30
        height_ratio = 0.34
        top_ratio = 0.18

    face_width = image_width * width_ratio
    face_height = min(image_height * height_ratio, face_width * 1.28)
    face_x = (image_width - face_width) / 2
    face_y = image_height * top_ratio

    max_face_y = max(0.0, image_height - face_height)
    face_y = min(face_y, max_face_y)

    return {
        "x": face_x,
        "y": face_y,
        "width": face_width,
        "height": face_height,
    }


def expand_face_to_head_box(face_box: dict, image_width: int, image_height: int) -> dict:
    x = face_box["x"]
    y = face_box["y"]
    w = face_box["width"]
    h = face_box["height"]

    head_x = max(0.0, x - w * DEFAULT_HEAD_SIDE_EXPANSION)
    head_y = max(0.0, y - h * DEFAULT_HEAD_TOP_EXPANSION)
    head_right = min(image_width, x + w + w * DEFAULT_HEAD_SIDE_EXPANSION)
    head_bottom = min(image_height, y + h + h * DEFAULT_HEAD_BOTTOM_EXPANSION)

    return {
        "x": head_x,
        "y": head_y,
        "width": max(1.0, head_right - head_x),
        "height": max(1.0, head_bottom - head_y),
    }


def compute_crop_box(image_size: tuple[int, int], face_box: dict | None, spec: OutputSpec) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    aspect_ratio = spec.width_px / spec.height_px

    if not face_box:
        face_box = estimate_fallback_face_box(image_width, image_height)

    fx = face_box["x"]
    fy = face_box["y"]
    fw = face_box["width"]
    fh = face_box["height"]
    head_box = expand_face_to_head_box(face_box, image_width, image_height)

    x = head_box["x"]
    y = head_box["y"]
    w = head_box["width"]
    h = head_box["height"]

    # Normalize by face size first so apparent head size stays consistent.
    crop_height = fh / spec.face_height_ratio
    crop_width = crop_height * aspect_ratio

    # Enlarge only as much as needed to keep the full expanded head box inside the crop.
    crop_height = max(crop_height, h / DEFAULT_HEAD_HEIGHT_RATIO)
    crop_width = max(crop_width, w * (1.0 + DEFAULT_HEAD_SIDE_EXPANSION))
    crop_height = crop_width / aspect_ratio

    max_crop_width, max_crop_height = fit_crop_within(image_width, image_height, aspect_ratio)
    crop_width = min(crop_width, max_crop_width)
    crop_height = min(crop_height, max_crop_height)
    if not math.isclose(crop_width / crop_height, aspect_ratio, rel_tol=0.001):
        crop_width = crop_height * aspect_ratio
        if crop_width > image_width:
            crop_width = image_width
            crop_height = crop_width / aspect_ratio

    face_center_x = fx + fw / 2
    head_center_y = y + h * DEFAULT_HEAD_ANCHOR_Y
    return clamp_crop(face_center_x, head_center_y, crop_width, crop_height, image_width, image_height)


def load_image(image_path: Path) -> np.ndarray:
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to decode image: {image_path}")
    return image


def final_extension(source_path: Path) -> str:
    if source_path.suffix.lower() in {".png", ".tif", ".tiff"}:
        return ".png"
    return ".jpg"


def crop_and_resize(
    image: np.ndarray,
    destination_path: Path,
    crop_box: tuple[int, int, int, int],
    spec: OutputSpec,
) -> None:
    left, top, right, bottom = crop_box
    cropped = image[top:bottom, left:right]
    resized = cv2.resize(cropped, (spec.width_px, spec.height_px), interpolation=cv2.INTER_LANCZOS4)

    ext = destination_path.suffix.lower()
    if ext == ".png":
        ok, encoded = cv2.imencode(".png", resized)
    else:
        ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"Unable to encode output image: {destination_path}")
    destination_path.write_bytes(encoded.tobytes())


def make_archive(output_dir: Path, archive_name: str) -> Path:
    archive_path = output_dir / archive_name
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path.name != archive_name:
                archive.write(path, arcname=path.name)
    return archive_path


def ensure_environment() -> None:
    if FACE_CASCADE.empty():
        raise RuntimeError(f"OpenCV face cascade is missing: {FACE_CASCADE_PATH}")


def process_batch(input_dir: Path, output_dir: Path, spec: OutputSpec) -> tuple[int, Path]:
    ensure_environment()
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input folder does not exist: {input_dir}")

    images = list_images(input_dir)
    if not images:
        raise ValueError(f"No supported image files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "Profile Photo Batcher",
        f"Input folder: {input_dir}",
        f"Output folder: {output_dir}",
        f"Requested output: {spec.width_px} x {spec.height_px} px",
        f"Units: {spec.units}",
        f"DPI: {spec.dpi}",
        f"Face fill: {spec.face_height_ratio:.2f}",
        "",
    ]

    processed_count = 0
    for image_path in images:
        try:
            image = load_image(image_path)
            image_size = (image.shape[1], image.shape[0])
            face = detect_primary_face(image_path)
            crop_box = compute_crop_box(image_size, face, spec)
            extension = final_extension(image_path)
            destination = output_dir / f"{image_path.stem}_{spec.width_px}x{spec.height_px}{extension}"
            crop_and_resize(image, destination, crop_box, spec)
            processed_count += 1
            mode = "face-detected" if face else "top-biased-fallback"
            report_lines.append(
                f"OK {image_path.name} -> {destination.name} | {mode} | crop={crop_box} | size={image_size}"
            )
        except Exception as exc:  # noqa: BLE001
            report_lines.append(f"FAILED {image_path.name}: {exc}")

    report_path = output_dir / "processing_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    archive_path = make_archive(output_dir, f"profile_photos_{spec.width_px}x{spec.height_px}.zip")
    return processed_count, archive_path


def main() -> int:
    try:
        input_dir, output_dir, spec = parse_args()
        count, archive_path = process_batch(input_dir, output_dir, spec)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Processed {count} photo(s).")
    print(f"Archive ready: {archive_path}")
    print(f"Report ready: {output_dir / 'processing_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
