FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
RUN sed -i "s@http://deb.debian.org@https://deb.debian.org@g" /etc/apt/sources.list.d/debian.sources && apt-get update && apt-get install -y --no-install-recommends bash curl git procps ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.2.6 Pillow==11.3.0
