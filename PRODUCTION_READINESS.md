# AR Survey Backend — Production Readiness

## Canonical repo decision

Current state: this standalone repo is treated as the **backend package**, while `/Users/tddud/ar-survey-inspection` is the broader inspection product repo/monorepo. Before launch, pick one canonical deployment source and stop hand-editing both.

Recommended: make `/Users/tddud/ar-survey-inspection/backend` canonical if Android + backend are shipped as one product, then archive or mirror this repo. Duplicate backend truth is a production footgun.

## Resolved today

- Production settings now reject weak/default `API_KEY`, `JWT_SECRET`, S3 keys and missing `ALLOWED_ORIGINS`.
- CORS is environment-driven via `ALLOWED_ORIGINS`.
- Upload init rejects non-video content types.
- Upload complete validates the storage object exists and size matches before processing.
- Upload completion is idempotent for jobs already processing/completed.
- Retry refuses to process if the video object has disappeared.
- Table auto-creation is development-only in this standalone backend.
- Added regression tests for production config and upload validation.

## Required environment variables for production

```bash
ENVIRONMENT=production
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
S3_ENDPOINT=https://...
S3_BUCKET=...
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
API_KEY=<temporary only until full RBAC is live>
JWT_SECRET=<32+ chars, generated secret>
ALLOWED_ORIGINS=https://app.example.com
BASE_URL=https://api.example.com
MAX_UPLOAD_BYTES=262144000
ALLOWED_UPLOAD_CONTENT_TYPES=video/mp4,video/quicktime,video/x-m4v
```

## Local verification

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Go/no-go gates still owned by Joseph/Kinetik

- Confirm canonical repo/deployment source.
- Provide production domains for `ALLOWED_ORIGINS` and `BASE_URL`.
- Provide production S3/Postgres/Redis credentials.
- Confirm target Android distribution route: APK sideload, private Play track, or MDM.
- Supply 20+ real representative inspection clips for AI quality baseline.
- Decide pilot users/sites and acceptance criteria.
