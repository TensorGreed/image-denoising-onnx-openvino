FROM rocm/pytorch:latest

WORKDIR /workspace

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        git \
        cmake && \
    rm -rf /var/lib/apt/lists/*

# Python deps (add facenet-pytorch)
RUN pip install --no-cache-dir \
        torchvision \
        matplotlib \
        ninja \
        facenet-pytorch

COPY train_face_denoiser.py .
COPY hip_addnoise ./hip_addnoise
COPY hip_linear ./hip_linear

WORKDIR /workspace/hip_addnoise
RUN python setup.py install

WORKDIR /workspace/hip_linear
RUN python setup.py install

WORKDIR /workspace

# Run face denoiser by default
CMD ["python", "train_face_denoiser.py", "--epochs", "5", "--batch-size", "32"]
