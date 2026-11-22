FROM robouniverse-sim:5.1.0-v1
COPY bridge/ /opt/roboenv/bridge/
COPY workspace/ /opt/roboenv/workspace/
COPY runtime/ /opt/roboenv/runtime/
COPY instance.json /opt/roboenv/instance.json
ENV PYTHONPATH=/opt/roboenv/extra:/opt/roboenv/bridge EMBODY_REPO_ROOT=/opt/roboenv/runtime EMBODY_TASK_WORKSPACE=/opt/roboenv/workspace OMNI_KIT_USER_CACHE=/tmp/roboenv-kit-cache
RUN mkdir -p /tmp/roboenv-kit-cache
WORKDIR /opt/roboenv/workspace
ENTRYPOINT []
CMD ["/opt/roboenv/bin/python", "-m", "embody_data.harbor.sidecar", "serve"]
