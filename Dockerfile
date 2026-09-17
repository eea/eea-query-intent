# eea-query-intent — query-intent classifier service (CPU-only).
#
# The trained model is fetched from Hugging Face AT BUILD TIME and baked into
# the image; the container needs no network access at runtime.
#
# Build (pin the revision to an immutable commit SHA for release builds):
#   docker build \
#     --build-arg HF_MODEL_REPO=eea/query-intent-setfit-v1 \
#     --build-arg HF_MODEL_REVISION=<commit-sha> \
#     -t eea-query-intent:1.0.0 .
#
# If the model repo is private, pass the token as a BuildKit secret instead
# of an ARG (never persist credentials in image metadata):
#   docker build --secret id=hf_token,env=HF_TOKEN ...
# and replace the snapshot RUN with:
#   RUN --mount=type=secret,id=hf_token \
#     HF_TOKEN="$(cat /run/secrets/hf_token)" python -c "..."
#
# Run:
#   docker run -p 8100:8100 eea-query-intent:1.0.0
#
# Runtime knobs (all optional):
#   EEA_QI_MODEL_PATH     default: /app/models/setfit
#   EEA_QI_DEVICE         default: cpu
#   EEA_QI_HOST           default: 0.0.0.0 in this image (loopback locally)
#   EEA_QI_PORT           default: 8100
#   EEA_QI_MAX_WORDS      default: 20
#   EEA_QI_ABSTAIN_THRESHOLD  default: value from the model's manifest.json
#
# The model repo must contain the SetFit weights, the tokenizer, and
# manifest.json (model_version, labels, eligible_labels, abstain_threshold).
# Pins match uv.lock.

FROM python:3.12-slim

ARG HF_MODEL_REPO=eea/query-intent-setfit-v1
# MUST be pinned to an immutable commit SHA for release builds; 'main' is
# only acceptable for throwaway test builds.
ARG HF_MODEL_REVISION=main

# CPU-only torch first, so pip never pulls the CUDA-bundled wheel (~2.5 GB).
RUN pip install --no-cache-dir torch==2.14.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi==0.141.1 \
    uvicorn==0.52.4 \
    setfit==1.2.0 \
    huggingface_hub==1.30.0 \
    numpy==1.26.4

WORKDIR /app
COPY src/ src/

# Bake the pinned model snapshot (weights + tokenizer + manifest.json) into
# the image. Runtime stays offline (HF_HUB_OFFLINE=1 below).
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${HF_MODEL_REPO}', revision='${HF_MODEL_REVISION}', local_dir='/app/models/setfit')"

ENV PYTHONUNBUFFERED=1 \
    EEA_QI_DEVICE=cpu \
    EEA_QI_HOST=0.0.0.0 \
    HF_HUB_OFFLINE=1

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=3)"
CMD ["python", "-m", "eea_query_intent.service"]
