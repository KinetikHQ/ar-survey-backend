# AR Survey & Inspection System

AI-powered construction site inspection using wearable cameras. Detects PPE compliance, safety hazards, quality defects, and progress status from short video clips.

**Internal tool for surveyors and site engineers.**

---

## Architecture

```
┌─────────────────────┐       ┌──────────────────────────────────────┐
│   Capture Device    │       │            Backend (Python)           │
│                     │       │                                      │
│  Android Emulator   │  ──►  │  ┌────────┐  ┌─────────┐  ┌───────┐ │
│  Meta Ray-Ban       │       │  │ FastAPI │──│ S3/     │──│ Worker│ │
│  Phone Camera       │  ◄──  │  │ Server  │  │ MinIO   │  │ (AI)  │ │
│                     │       │  └────────┘  └─────────┘  └───────┘ │
│  Local Queue        │       │       │                              │
│  Retry Engine       │       │  ┌─────────┐  ┌─────────┐          │
│                     │       │  │ Redis   │  │ SQLite/ │          │
│                     │       │  │ Queue   │  │ Postgres│          │
│                     │       │  └─────────┘  └─────────┘          │
└─────────────────────┘       └──────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|---|---|
| API Server | Python, FastAPI |
| Database | SQLite (dev), PostgreSQL (prod) |
| Job Queue | Redis + RQ (fallback: threading for dev) |
| Object Storage | S3/MinIO (prod), local filesystem (dev) |
| AI Detection | YOLOv8n (person detection) + heuristic PPE classifiers |
| Frame Extraction | OpenCV (1 fps sampling) |
| Android App | Kotlin, Android 10+ |
| Target Hardware | Phone camera (MVP), Meta Ray-Ban Display glasses (later) |

---

## Quick Start

### Prerequisites

- Python 3.9+
- Git

### Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r ai/requirements-ai.txt
cp .env.example .env
```

### Run

```bash
# Start the API server
cd backend
source venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API starts with:
- SQLite database (no Docker needed)
- Local file storage (no S3/MinIO needed)
- Inline processing (no Redis needed)

Visit **http://localhost:8000/docs** for interactive API documentation.

### Test

```bash
# Full pipeline test with demo video (creates simulated PPE scene)
python scripts/test_upload.py --demo

# Upload your own video
python scripts/test_upload.py path/to/video.mp4

# Point at a remote server
python scripts/test_upload.py video.mp4 --api-url http://192.168.1.100:8000
```

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | No | Health check |
| `POST` | `/api/v1/upload/init` | Yes | Create job, get upload URL |
| `POST` | `/api/v1/upload/complete` | Yes | Mark upload done, start processing |
| `GET` | `/api/v1/jobs/{job_id}` | Yes | Get job status and results |
| `POST` | `/api/v1/jobs/{job_id}/retry` | Yes | Retry a failed job |

All authenticated endpoints require: `Authorization: Bearer <API_KEY>`

### Upload Flow

```
1. POST /upload/init     → Get job_id + presigned upload URL
2. PUT video to URL       → Direct upload (S3 or local dev endpoint)
3. POST /upload/complete  → Triggers AI processing
4. Poll /jobs/{job_id}    → Check status every 5s
5. Status = completed     → Read results
```

### Response Format

```json
{
  "job_id": "uuid",
  "status": "completed",
  "results": [
    {
      "id": "uuid",
      "category": "ppe",
      "label": "missing_hard_hat",
      "confidence": 0.94,
      "bbox": [120, 80, 340, 280],
      "frame_timestamp": 12.5,
      "metadata": {"description": "Person detected without hard hat"}
    }
  ],
  "summary": {
    "total_detections": 2,
    "ppe_violations": 1,
    "ppe_compliant": 1,
    "frames_analyzed": 45
  }
}
```

### Detection Labels (MVP — PPE)

| Label | Meaning |
|---|---|
| `hard_hat_present` | Person wearing hard hat |
| `missing_hard_hat` | Person without hard hat |
| `hi_vis_present` | Person wearing hi-vis vest |
| `missing_hi_vis` | Person without hi-vis vest |

---

## Testing

### Option 1: CLI Test Harness (No App Needed)

The fastest way to test the full pipeline. Uploads a video, processes it, and prints results.

```bash
# Create a demo video with simulated PPE scene and process it
python scripts/test_upload.py --demo

# Process a real video
python scripts/test_upload.py recording.mp4

# Poll an existing job
python scripts/test_upload.py --watch <job_id>

# Custom API endpoint
python scripts/test_upload.py video.mp4 --api-url http://10.0.0.5:8000
```

### Option 2: Android Emulator (Meta Mock Device Kit)

For testing the full Android app flow without physical glasses.

#### Setup

1. **Install Android Studio** — https://developer.android.com/studio

2. **Create an Android Virtual Device (AVD):**
   - Open Android Studio → Device Manager → Create Device
   - Select **Pixel 6** or similar
   - System image: **API 33+ (Android 13+)**
   - Finish and launch

3. **Configure Meta Mock Device Kit:**
   - Download from [Meta Wearables Developer Center](https://developer.meta.com/)
   - Install the Mock Device Kit APK on the emulator:
     ```bash
     adb install mock-device-kit.apk
     ```
   - The Mock Device Kit simulates Ray-Ban Display glasses — camera, display, sensors

4. **Point the app at local backend:**
   - In the Android app settings, set API URL to:
     ```
     http://10.0.2.2:8000
     ```
     (`10.0.2.2` is the emulator's alias for the host machine's localhost)

#### Testing Flow

```
1. Start backend:     uvicorn api.main:app --host 0.0.0.0 --port 8000
2. Launch emulator:   Android Studio → Run → Select device
3. Open app:          Record a clip → Upload → View results
4. Check backend:     curl http://localhost:8000/api/v1/jobs/<job_id>
```

### Option 3: Meta Ray-Ban Display Glasses

For testing with real hardware on a construction site.

#### Prerequisites

- Meta developer account (https://developer.meta.com/)
- Physical Ray-Ban Display glasses
- App signed and installed on glasses companion device

#### Setup

1. **Deploy backend to accessible network:**
   ```bash
   # On a laptop connected to the same network as the glasses
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   
   # Or deploy to a cloud VM
   # Note: requires PostgreSQL + Redis + S3 in production
   docker compose up -d
   ```

2. **Configure app with server URL:**
   - Local network: `http://<laptop-ip>:8000`
   - Cloud: `https://your-server.com`

3. **Test on-site:**
   - Wear glasses → Record clip → App uploads → Results display on glasses

#### Network Considerations

| Connection | Behaviour |
|---|---|
| Wi-Fi | Immediate upload |
| Cellular | Upload with data saver check |
| Offline | Clips saved locally, upload on reconnect |

---

## Development

### Project Structure

```
ar-survey-inspection/
├── backend/
│   ├── api/
│   │   ├── routes.py          # All API endpoints
│   │   ├── auth.py            # Bearer token authentication
│   │   └── main.py            # FastAPI app entry point
│   ├── ai/
│   │   ├── pipeline.py        # Orchestrator: extract → detect → deduplicate
│   │   └── detectors/
│   │       ├── ppe_detector.py    # YOLOv8n + heuristic PPE classification
│   │       └── frame_extractor.py # OpenCV frame extraction (1 fps)
│   ├── models/
│   │   ├── database.py        # SQLAlchemy models (Job, Result)
│   │   └── session.py         # Database session factory
│   ├── workers/
│   │   └── processor.py       # RQ worker for AI processing
│   ├── storage.py             # S3/MinIO + local file fallback
│   ├── config.py              # Environment settings
│   ├── requirements.txt       # Python dependencies
│   └── .env.example           # Environment template
├── android/                   # Android app (developed separately)
├── scripts/
│   └── test_upload.py         # CLI test harness
├── docs/
│   ├── day-1-summary.md
│   └── day-2-summary.md
├── docker-compose.yml         # Full stack (Postgres, Redis, MinIO)
└── README.md
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./ar_survey.db` | Database connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection (optional in dev) |
| `S3_ENDPOINT` | `http://localhost:9000` | S3/MinIO endpoint (optional in dev) |
| `S3_BUCKET` | `ar-survey-clips` | S3 bucket name |
| `S3_ACCESS_KEY` | — | S3 access key |
| `S3_SECRET_KEY` | — | S3 secret key |
| `API_KEY` | `dev-api-key-change-in-prod` | Bearer token for API auth |
| `MODEL_DEVICE` | `cpu` | `cpu` or `cuda` for GPU inference |
| `FRAME_SAMPLE_RATE` | `1` | Frames per second to analyse |

### Running with Docker (Full Stack)

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f api

# Stop
docker compose down
```

Docker services: API (`:8000`), Worker, PostgreSQL (`:5432`), Redis (`:6379`), MinIO (`:9000`)

### Running Tests

```bash
cd backend
source venv/bin/activate

# Full test suite
pytest tests/ -v

# API tests only
pytest tests/test_api.py -v

# AI pipeline tests only
pytest tests/test_pipeline.py -v
```

### AI Pipeline Details

The detection pipeline processes video clips in 4 stages:

```
Video File
    │
    ▼
Frame Extraction (OpenCV, 1 fps)
    │
    ▼
YOLOv8n Person Detection (COCO pre-trained)
    │
    ▼
Heuristic PPE Classification
  ├── Hard hat: bright dome detection in upper 25% of person ROI
  └── Hi-vis: fluorescent colour band detection in torso 30-70%
    │
    ▼
Deduplication (IoU > 0.5 within 2s window)
    │
    ▼
Results (JSON)
```

**CPU-first:** Uses `yolov8n.pt` (nano, ~6MB) for fast inference on CPU. Switch to `yolov8m.pt` or `yolov8l.pt` for higher accuracy on GPU.

---

## Deployment

### Development (No Docker)

Works out of the box with SQLite + local file storage. No external services needed.

```bash
cd backend
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Production (Docker)

```bash
docker compose up -d
```

Requires:
- PostgreSQL for job/result storage
- Redis for job queue
- S3-compatible storage (AWS S3, MinIO, etc.)
- GPU worker for faster inference (optional)

### Production Checklist

- [ ] Change `API_KEY` to a strong random value
- [ ] Set `DATABASE_URL` to PostgreSQL
- [ ] Set `MODEL_DEVICE=cuda` if GPU available
- [ ] Configure S3 credentials
- [ ] Set up HTTPS (nginx/caddy reverse proxy)
- [ ] Deploy behind VPN for internal access

---

## Roadmap

### MVP (Current)
- [x] FastAPI backend with job queue
- [x] YOLOv8n PPE detection (hard hat, hi-vis)
- [x] CLI test harness
- [x] Dev mode (no Docker/S3/Redis needed)
- [ ] Android app with camera capture
- [ ] Offline-first queue on device

### Post-MVP
- [ ] Meta Ray-Ban Display integration
- [ ] SAM 3.1 for precise segmentation
- [ ] Crack and defect detection
- [ ] Progress status classification
- [ ] GPU worker scaling
- [ ] Multi-language support

---

## Contributing

This is an internal Kinetik project. Standard workflow:

1. Create a branch from `main`
2. Make changes
3. Run tests
4. Submit PR for review

---

## License

Internal use only. Kinetik © 2026.
