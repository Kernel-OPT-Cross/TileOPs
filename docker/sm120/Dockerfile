FROM nvidia/cuda:13.2.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MINICONDA_VERSION=py312_25.9.1-3

ENV PATH=/opt/conda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL \
       "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" \
       -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

WORKDIR /workspace/TileOPs
COPY . .

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e . \
    && python -m pip install --group dev \
    && python -m pip check

CMD ["bash"]
