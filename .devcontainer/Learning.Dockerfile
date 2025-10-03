# ---------- Base Python ----------
    FROM python:3.12-slim AS base

    # Install system dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        curl \
        ca-certificates \
        cmake \
        bison \
        flex \
        pkg-config \
        g++ \
        make \
        libgmp-dev \
        zlib1g-dev \
        && rm -rf /var/lib/apt/lists/*

    # ---------- Build Spot ----------
    FROM base AS spot-builder
    WORKDIR /src

    ENV SPOT_VERSION=2.14.1
    ENV SPOT_SHA256=25df8a6af4e4bb3ae67515ac98e3d37c4303a682e33aaa66e72d74b39459a530

    RUN wget http://www.lre.epita.fr/dload/spot/spot-${SPOT_VERSION}.tar.gz \
        && echo "${SPOT_SHA256}  spot-${SPOT_VERSION}.tar.gz" | sha256sum -c - \
        && tar xzf spot-${SPOT_VERSION}.tar.gz \
        && cd spot-${SPOT_VERSION} \
        && ./configure --prefix=/usr/local --enable-python --enable-tools \
        && make -j"$(nproc)" \
        && make install \
        && ldconfig

    # ---------- Build SyFCo ----------
    FROM debian:bookworm AS syfco-builder
    WORKDIR /src

    # Install Haskell, Cabal, and git
    RUN apt-get update && apt-get install -y --no-install-recommends \
        ghc \
        cabal-install \
        git \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

    RUN git clone --depth 1 https://github.com/reactive-systems/syfco.git \
        && cd syfco \
        && cabal update \
        && cabal v2-install --installdir=/usr/local/bin --overwrite-policy=always

    # ---------- Final Image ----------
    FROM base

    # Copy Spot from builder
    COPY --from=spot-builder /usr/local /usr/local

    # Copy SyFCo binary
    COPY --from=syfco-builder /usr/local/bin/syfco /usr/local/bin/syfco

    # Set working directory
    WORKDIR /workdir

    # Copy your project into the container
    COPY . /workdir/Automata_SSM_Learning

    # Install PyTorch 2.7.1 + deps
    RUN pip install --no-cache-dir torch==2.7.1 \
        && pip install --no-cache-dir \
           jinja2 sympy networkx setuptools typing-extensions \
           torchvision torchaudio torchtext torchmetrics

    # Default command
    CMD ["/bin/bash"]
    CMD ["sleep", "infinity"]
