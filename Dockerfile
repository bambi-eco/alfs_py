FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y xvfb

# Install dependencies
RUN pip install hatch
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

ENV DISPLAY=:99
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

# Copy everything at once to maintain the project structure
COPY . /app/

RUN pip install .

# Run the script
ENTRYPOINT ["sh", "-c", "Xvfb :99 -screen 0 1024x768x24+32 & sleep 2 && python src/alfspy/orthografic_projection.py"]
