FROM robouniverse-agent:1
COPY roboenv_client.py /opt/roboenv/
COPY roboenv /usr/local/bin/roboenv
RUN chmod 755 /usr/local/bin/roboenv
ENV PYTHONPATH=/opt/roboenv
WORKDIR /app
CMD ["sleep", "infinity"]
