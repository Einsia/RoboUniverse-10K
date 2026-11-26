FROM nvcr.io/nvidia/isaac-sim:5.1.0@sha256:f3563cb2ba0c18af0b2fb321360dcb73a917b899f879e3213623d6bee484fa54
USER root
COPY python-wrapper /opt/roboenv/bin/python
RUN chmod 755 /opt/roboenv/bin/python && mkdir -p /opt/roboenv/runtime/.venvs/sim/bin && ln -s /opt/roboenv/bin/python /opt/roboenv/runtime/.venvs/sim/bin/python
RUN /isaac-sim/python.sh -m pip install --no-cache-dir --target /opt/roboenv/extra imageio-ffmpeg==0.8.1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility OMNI_KIT_ALLOW_ROOT=1
ENTRYPOINT []
