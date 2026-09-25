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

# CPU-only torch first, so pip never pulls the CUDA-bundled wheel
# (~2.5 GB). --only-binary=:all: refuses sdist builds (no setup-script
# execution, S8541). Every package - torch plus all of its transitive
# deps - is pinned to the exact version the CI build resolved (S8544,
# locked versions); pip still verifies the full graph, so a stale pin
# fails the build loudly.
RUN pip install --no-cache-dir --only-binary=:all: \
    torch==2.14.0+cpu \
    filelock==3.32.3 \
    fsspec==2026.6.0 \
    jinja2==3.1.6 \
    markupsafe==3.0.3 \
    mpmath==1.3.0 \
    networkx==3.6.1 \
    setuptools==78.1.0 \
    sympy==1.14.0 \
    typing-extensions==4.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Split from the torch layer on purpose (S7031 waived): torch is a big
# rarely-changing layer; keeping the small deps separate preserves its
# cache. All 62 packages - the five direct deps plus every transitive -
# are pinned exactly (S8544); --only-binary=:all: (S8541). The fsspec
# pin matches layer 1 so the build never downgrades mid-image.
RUN pip install --no-cache-dir --only-binary=:all: \
    accelerate==1.15.0 \
    aiohappyeyeballs==2.7.1 \
    aiohttp==3.14.3 \
    aiosignal==1.4.0 \
    annotated-doc==0.0.5 \
    annotated-types==0.8.0 \
    anyio==4.15.1 \
    attrs==26.1.0 \
    certifi==2026.7.22 \
    charset-normalizer==3.5.1 \
    click==8.5.0 \
    cloudpickle==3.1.2 \
    datasets==5.0.1 \
    dill==0.4.1 \
    evaluate==0.4.6 \
    fastapi==0.141.1 \
    frozenlist==1.8.0 \
    fsspec==2026.6.0 \
    h11==0.16.0 \
    hf-xet==1.6.0 \
    httpcore==1.0.9 \
    httpx==0.28.1 \
    huggingface-hub==1.30.0 \
    idna==3.20 \
    joblib==1.6.0 \
    markdown-it-py==4.2.0 \
    mdurl==0.1.2 \
    multidict==6.9.1 \
    multiprocess==0.70.19 \
    narwhals==2.26.0 \
    numpy==1.26.4 \
    packaging==26.3 \
    pandas==3.0.6 \
    propcache==0.5.4 \
    psutil==7.2.2 \
    pyarrow==25.0.1 \
    pydantic==2.13.5 \
    pydantic-core==2.46.5 \
    pygments==2.21.0 \
    python-dateutil==2.9.0.post0 \
    pyyaml==6.0.3 \
    regex==2026.9.10 \
    requests==2.34.2 \
    rich==15.0.0 \
    safetensors==0.8.0 \
    scikit-learn==1.9.1 \
    scipy==1.17.1 \
    sentence-transformers==6.1.0 \
    setfit==1.2.0 \
    shellingham==1.5.4 \
    six==1.17.0 \
    starlette==1.7.0 \
    threadpoolctl==3.7.0 \
    tokenizers==0.23.2 \
    tqdm==4.70.1 \
    transformers==5.17.0 \
    typer==0.27.2 \
    typing-inspection==0.4.4 \
    urllib3==2.8.0 \
    uvicorn==0.52.4 \
    xxhash==4.0.1 \
    yarl==1.25.1

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
