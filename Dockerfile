# eea-query-intent — query-intent classifier service (CPU-only).
#
# The trained model is fetched from Hugging Face AT BUILD TIME and baked into
# the image; the container needs no network access at runtime.
#
# Build (pin the revision to an immutable commit SHA for release builds):
#   docker build \
#     -t eea-query-intent:1.0.0 .   # repo + revision already pinned below
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

ARG HF_MODEL_REPO=eeahugs/query-intent-setfit-v1
# Pinned to the immutable push commit of eeahugs/query-intent-setfit-v1
# (2026-09-18). Change only for a deliberately new model version.
ARG HF_MODEL_REVISION=ba111788b58e4f1e0dafb3e74189f0a08d1e3186

# Runtime lockfile (S8544): every package is pinned exactly in this file,
# which is the single source of truth for the image's dependencies.
COPY runtime-requirements.txt /tmp/runtime-requirements.txt

# Single install layer (S7031): bake a pip config (wheel-only installs,
# no sdist builds - S8541 hardening; and the PyTorch CPU mirror as an
# extra index, from which the +cpu torch pin is the only resolvable
# source, so the CUDA-bundled wheel is never pulled) and install the
# locked file. pip still verifies the full dependency graph, so a stale
# pin fails the build loudly.
RUN printf '[global]\n' > /etc/pip.conf \
    && printf 'only-binary = :all:\n' >> /etc/pip.conf \
    && printf 'extra-index-url = https://download.pytorch.org/whl/cpu\n' >> /etc/pip.conf \
    && pip install --no-cache-dir -r /tmp/runtime-requirements.txt

WORKDIR /app
COPY src/ src/

# Bake the pinned model snapshot (weights + tokenizer + manifest.json) into
# the image. Runtime stays offline (HF_HUB_OFFLINE=1 below). The whole
# -c program is one double-quoted shell string (Dockerfile-level
# continuations) so the ARG values are expanded by the shell.
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='${HF_MODEL_REPO}', \
    revision='${HF_MODEL_REVISION}', local_dir='/app/models/setfit')"

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    EEA_QI_DEVICE=cpu \
    EEA_QI_HOST=0.0.0.0 \
    HF_HUB_OFFLINE=1

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=3)"
CMD ["python", "-m", "eea_query_intent.service"]
