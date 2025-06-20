FROM postgres:16


# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libreadline-dev \
    zlib1g-dev flex \
    bison \
    postgresql-server-dev-16 \
    && rm -rf /var/lib/apt/lists/*

# Clone and build Apache AGE
RUN git clone https://github.com/apache/age.git /age

WORKDIR /age
RUN make install

# Cleanup
WORKDIR /
RUN rm -rf /age

CMD ["postgres", "-c", "shared_preload_libraries=age"]
