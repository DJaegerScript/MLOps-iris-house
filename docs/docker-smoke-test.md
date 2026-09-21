# Docker smoke test

The image contains the Streamlit application and manifest metadata only. The
model bundle remains in the private, versioned S3 object configured at runtime.

Build the image, start Streamlit as the image's non-root user, verify that
`8501/tcp` is exposed, and check `/_stcore/health` without loading a model:

```bash
python scripts/docker_smoke_test.py
```

To submit a real sample prediction, provide all three S3 coordinates and an
AWS profile whose read permissions are limited to that object. The profile is
mounted read-only at runtime and is never copied into the image:

```bash
python scripts/docker_smoke_test.py \
  --sample-prediction \
  --aws-profile iris-mlops-readonly \
  --model-s3-bucket YOUR_PRIVATE_BUCKET \
  --model-s3-key models/iris-classifier-v1.tar.gz \
  --model-s3-version-id YOUR_S3_VERSION_ID
```

Install Playwright and its browser only on the machine running the optional UI
check:

```bash
python -m pip install playwright
python -m playwright install chromium
```

The harness never accepts or forwards static AWS access-key values. ECS should
use its task role instead of an AWS profile mount.
