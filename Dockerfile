# ALFS renderer, PyTorch backend.
#
# The ModernGL image needed libgl1-mesa-glx, the X11 dev headers and an Xvfb virtual display,
# and it was that software-GL-under-Xvfb combination that produced the intermittent
# transparency artifacts documented in docs/MIGRATION_REVIEW.md. None of it is needed now:
# there is no GL driver, no X server and no DISPLAY.
FROM python:3.11-slim

WORKDIR /app

# OpenCV needs libGL for its own image codecs even in headless builds, plus libglib.
# This is *not* a rendering dependency -- nothing here opens a GL context.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install the CPU torch build explicitly. The default PyPI wheel pulls the full CUDA runtime
# (~2 GB) even when no GPU is present. For GPU deployments, drop the --index-url line and
# rebuild, or base the image on an nvidia/cuda runtime image.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV INPUT_DIR=/input
ENV OUTPUT_DIR=/output
ENV SPLITS=train,val,test
ENV CAMERA_DISTANCE=10.0
ENV ORTHO_WIDTH=70
ENV ORTHO_HEIGHT=70
ENV INPUT_WIDTH=1024
ENV INPUT_HEIGHT=1024
ENV RENDER_WIDTH=2048
ENV RENDER_HEIGHT=2048
ENV INITIAL_SKIP=0
ENV ADD_BACKGROUND=1
ENV FOVY=50.0
ENV ASPECT_RATIO=1.0
ENV SAVE_LABELED_IMAGES=0
ENV ADDITIONAL_ROTATIONS=0
ENV ROTATION_LIMIT=6.28318530718
ENV ROTATION_SEED=-1
ENV ROTATION_LIMIT_RADIAN=1
ENV EXCLUDE_FLIGHTS=""
ENV MERGE_LABELS_IN_ALFS=1
ENV APPLY_NMS=0
ENV NMS_IOU=0.9
ENV IS_THERMAL=1
ENV USE_ONEFILE_CORRECTIONS=1

# Selects the torch device. Leave unset to use CUDA when visible, otherwise CPU.
ENV ALFS_DEVICE=""

# Copy everything at once to maintain the project structure
COPY . /app/

RUN pip install --no-cache-dir .

# Sanity-check that the render backend imports and can rasterise, so a broken image fails at
# build time rather than halfway through a dataset.
RUN python -c "import alfspy.core.torchgl as t; print('torchgl ok', t.TORCH_VERSION, t.COMPAT_NOTES)"

# No Xvfb, no DISPLAY, no sleep-and-hope startup.
ENTRYPOINT ["python", "src/alfspy/orthografic_projection.py"]
