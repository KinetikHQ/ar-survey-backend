#!/usr/bin/env python3
"""AR Survey Inspection — CLI Test Harness

Send video files to the API for processing. Works with any MP4.

Usage:
    python scripts/test_upload.py video.mp4
    python scripts/test_upload.py video.mp4 --api-url http://192.168.1.100:8000
    python scripts/test_upload.py --demo              # Create and upload dummy video
    python scripts/test_upload.py --watch job_id       # Poll an existing job
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

DEFAULT_API = "http://127.0.0.1:8000"
DEFAULT_API_KEY = "dev-api-key-change-in-prod"


def create_dummy_video(path: str, duration: int = 10, fps: int = 30) -> str:
    """Create a test video with some coloured shapes (simulates a scene)."""
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (640, 480))

    for frame_num in range(duration * fps):
        t = frame_num / fps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Background — concrete grey
        frame[:, :] = (160, 160, 160)

        # Simulated "person" — stick figure
        cx = 200 + int(50 * np.sin(t))
        cy = 240
        # Head
        cv2.circle(frame, (cx, cy - 80), 25, (220, 180, 150), -1)
        # Body
        cv2.line(frame, (cx, cy - 55), (cx, cy + 40), (220, 180, 150), 3)
        # Arms
        cv2.line(frame, (cx, cy - 30), (cx - 40, cy), (220, 180, 150), 3)
        cv2.line(frame, (cx, cy - 30), (cx + 40, cy), (220, 180, 150), 3)
        # Legs
        cv2.line(frame, (cx, cy + 40), (cx - 30, cy + 100), (220, 180, 150), 3)
        cv2.line(frame, (cx, cy + 40), (cx + 30, cy + 100), (220, 180, 150), 3)

        # Hard hat (yellow dome on head)
        cv2.ellipse(frame, (cx, cy - 100), (30, 15), 0, 180, 360, (0, 220, 255), -1)

        # Second "person" without hard hat
        cx2 = 450 + int(30 * np.cos(t * 0.7))
        cv2.circle(frame, (cx2, cy - 80), 25, (200, 160, 130), -1)
        cv2.line(frame, (cx2, cy - 55), (cx2, cy + 40), (200, 160, 130), 3)
        cv2.line(frame, (cx2, cy - 30), (cx2 - 40, cy), (200, 160, 130), 3)
        cv2.line(frame, (cx2, cy - 30), (cx2 + 40, cy), (200, 160, 130), 3)
        cv2.line(frame, (cx2, cy + 40), (cx2 - 30, cy + 100), (200, 160, 130), 3)
        cv2.line(frame, (cx2, cy + 40), (cx2 + 30, cy + 100), (200, 160, 130), 3)

        # Hi-vis vest on person 1 (fluorescent yellow band)
        cv2.rectangle(frame, (cx - 20, cy - 50), (cx + 20, cy + 10), (0, 230, 230), -1)

        writer.write(frame)

    writer.release()
    return path


def upload_video(api_url: str, api_key: str, video_path: str) -> dict:
    """Full upload flow: init → upload → complete → poll results."""
    headers = {"Authorization": f"Bearer {api_key}"}
    path = Path(video_path)

    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    file_size = path.stat().st_size
    print(f"Uploading: {path.name} ({file_size / 1024 / 1024:.1f} MB)")

    # 1. Init
    print("1. Creating job...")
    r = requests.post(
        f"{api_url}/api/v1/upload/init",
        headers=headers,
        json={
            "device_id": "test-harness-001",
            "duration_seconds": 10,
            "content_type": "video/mp4",
        },
    )
    r.raise_for_status()
    data = r.json()
    job_id = data["job_id"]
    print(f"   Job ID: {job_id}")

    # 2. Upload (dev mode: POST multipart to local endpoint)
    print("2. Uploading video...")
    upload_url = data["upload_url"]
    if "localhost:8000" in upload_url or "127.0.0.1:8000" in upload_url:
        # Dev mode — POST to local endpoint
        with open(path, "rb") as f:
            r = requests.post(
                f"{api_url}/api/v1/dev/upload-file/{job_id}",
                files={"file": (path.name, f, "video/mp4")},
            )
        r.raise_for_status()
        print(f"   Saved: {r.json()['path']}")
    else:
        # Production — PUT to presigned URL
        with open(path, "rb") as f:
            r = requests.put(
                upload_url,
                data=f,
                headers={"Content-Type": "video/mp4"},
            )
        r.raise_for_status()
        print(f"   Uploaded to S3")

    # 3. Mark complete
    print("3. Marking upload complete...")
    r = requests.post(
        f"{api_url}/api/v1/upload/complete",
        headers=headers,
        json={"job_id": job_id, "file_size_bytes": file_size},
    )
    r.raise_for_status()
    print(f"   Status: {r.json()['status']}")

    return {"job_id": job_id}


def poll_results(api_url: str, api_key: str, job_id: str, max_wait: int = 120) -> dict:
    """Poll until job completes or fails."""
    headers = {"Authorization": f"Bearer {api_key}"}
    print(f"4. Polling job {job_id[:8]}...")

    start = time.time()
    while time.time() - start < max_wait:
        r = requests.get(f"{api_url}/api/v1/jobs/{job_id}", headers=headers)
        r.raise_for_status()
        result = r.json()
        status = result["status"]

        if status in ("completed", "failed"):
            elapsed = time.time() - start
            print(f"   Done in {elapsed:.1f}s — {status}")
            return result

        time.sleep(2)

    print(f"   Timeout after {max_wait}s")
    return result


def print_results(result: dict):
    """Pretty-print detection results."""
    print()
    if result.get("error_message"):
        print(f"ERROR: {result['error_message']}")
        return

    if result.get("summary"):
        s = result["summary"]
        print(f"Summary: {s['total_detections']} detections, "
              f"{s['ppe_violations']} violations, "
              f"{s['ppe_compliant']} compliant, "
              f"{s['frames_analyzed']} frames")

    if result.get("results"):
        print()
        for det in result["results"]:
            icon = "🔴" if "missing" in det["label"] else "🟢"
            print(f"  {icon} {det['label']:20s}  conf={det['confidence']:.3f}  "
                  f"@ {det['frame_timestamp']:.1f}s  bbox={det['bbox']}")
    else:
        print("No detections (empty/blank video?)")


def main():
    parser = argparse.ArgumentParser(description="AR Survey Test Harness")
    parser.add_argument("video", nargs="?", help="Path to video file (.mp4)")
    parser.add_argument("--api-url", default=DEFAULT_API, help="API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--demo", action="store_true", help="Create and upload dummy video")
    parser.add_argument("--watch", metavar="JOB_ID", help="Poll an existing job")
    parser.add_argument("--timeout", type=int, default=120, help="Max wait seconds")
    args = parser.parse_args()

    if args.watch:
        result = poll_results(args.api_url, args.api_key, args.watch, args.timeout)
        print_results(result)
        return

    if args.demo:
        dummy_path = "/tmp/ar_survey_demo.mp4"
        print("Creating demo video with simulated PPE scene...")
        create_dummy_video(dummy_path, duration=10)
        print(f"Created: {dummy_path}")
        video_path = dummy_path
    elif args.video:
        video_path = args.video
    else:
        parser.print_help()
        sys.exit(1)

    # Upload and process
    result = upload_video(args.api_url, args.api_key, video_path)
    final = poll_results(args.api_url, args.api_key, result["job_id"], args.timeout)
    print_results(final)


if __name__ == "__main__":
    main()
