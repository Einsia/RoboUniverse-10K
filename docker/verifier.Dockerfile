FROM robouniverse-agent:1
COPY test.sh grade.py /tests/
WORKDIR /tests
