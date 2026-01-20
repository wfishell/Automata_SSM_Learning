# ---------- Base image for all stages ----------
FROM public.ecr.aws/lambda/python:3.11 AS base

# ---------- Build Spot ----------
FROM base AS spot-builder
WORKDIR /src

RUN yum install -y gcc gcc-c++ make wget tar gzip bison flex

ENV SPOT_VERSION=2.14.1
RUN wget http://www.lre.epita.fr/dload/spot/spot-${SPOT_VERSION}.tar.gz \
    && tar xzf spot-${SPOT_VERSION}.tar.gz \
    && cd spot-${SPOT_VERSION} \
    && ./configure --prefix=/usr/local --disable-python \
    && make -j"$(nproc)" \
    && make install

# ---------- Build SyFCo ----------
FROM base AS syfco-builder
WORKDIR /src

RUN yum install -y gcc gcc-c++ gmp-devel make wget tar gzip git ncurses-devel xz

RUN ln -sf /usr/bin/make /usr/bin/gmake

# Install GHCup and GHC
RUN curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | \
    BOOTSTRAP_HASKELL_NONINTERACTIVE=1 \
    BOOTSTRAP_HASKELL_GHC_VERSION=9.2.8 \
    BOOTSTRAP_HASKELL_CABAL_VERSION=3.6.2.0 \
    sh

ENV PATH="/root/.ghcup/bin:$PATH"

RUN git clone --depth 1 https://github.com/reactive-systems/syfco.git \
    && cd syfco \
    && cabal update \
    && cabal v2-install --allow-newer=base --installdir=/usr/local/bin --overwrite-policy=always

# ---------- Final Lambda Image ----------
FROM base

# Copy Spot binaries and libraries
COPY --from=spot-builder /usr/local/bin/ltlsynt /usr/local/bin/
COPY --from=spot-builder /usr/local/bin/autfilt /usr/local/bin/
COPY --from=spot-builder /usr/local/lib/libspot*.so* /usr/local/lib/
COPY --from=spot-builder /usr/local/lib/libbddx*.so* /usr/local/lib/

# Copy SyFCo binary
COPY --from=syfco-builder /usr/local/bin/syfco /usr/local/bin/syfco

# Add libraries to path
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/bin:$PATH

# Install Python dependencies
RUN pip install --only-binary=:all: pandas boto3 aalpy pydot

# Copy handler and scripts
COPY handler.py ${LAMBDA_TASK_ROOT}/
COPY active_learning.py ${LAMBDA_TASK_ROOT}/
COPY Dot_Trace_Generator.py ${LAMBDA_TASK_ROOT}/
COPY Trace_Checker.py ${LAMBDA_TASK_ROOT}/
COPY handler_passive.py ${LAMBDA_TASK_ROOT}/
COPY Passive_Mealy_Learning.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
