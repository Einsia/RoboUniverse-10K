# RoboUniverse evaluation

Run a coding agent or a reference controller against one physical manipulation environment or the frozen 500-task selection. A Docker Compose simulation service owns the scene and evidence. The agent gets the task instruction and an observation/action client. Harbor runs the verifier separately after collection. A successful verdict earns 1, failure or inconclusive earns 0; an infrastructure exception is not a scored attempt.

Use Linux x86_64, an RTX-capable NVIDIA GPU/driver compatible with Isaac Sim 5.1, NVIDIA Container Toolkit, Docker and Compose v2+, Python 3.12, and disk space for the Isaac installation, task builds and trial logs. Allow outbound HTTPS to Python, Debian, NVIDIA, PyTorch and asset sources during setup. Evaluation requires the Isaac Sim license; review it before running.

From this repository:

~~~bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# Set DATA to a local HF dataset snapshot, not the GitHub source tree.
# e.g. hf download "$HF_DATASET" --repo-type dataset --revision "$HF_REVISION" --local-dir ./dataset
export DATA=./dataset
python evaluate.py list --data "$DATA" --split heldout500
docker build -t robouniverse-sim:5.1.0-v1 docker/sim-base
docker build -t robouniverse-agent:1 docker/agent-base
python evaluate.py run --data "$DATA" --case 000002_door_hinge_or_open_close --work ./work/oracle --agent oracle --gpu 0 --accept-isaac-eula
~~~

To evaluate a model, set the provider credentials required by its Harbor adapter in your shell, then run:

~~~bash
python evaluate.py run --data "$DATA" --split heldout500 --work ./work/agent-trial --agent "$HARBOR_AGENT" --model "$HARBOR_MODEL" --gpu 0 --accept-isaac-eula
~~~

Each run requires a new work directory. Run one agent per task, sequentially, with no automatic retries. Keep work directories private: they contain ephemeral access tokens, logs and captures. The normal results are under work/agent-trial/jobs/CASE; a verifier result is not inferred from the CLI exit status. Use summarize.py to join trial outcomes to the intended cases and report missing and error counts separately. For storage-constrained runs, download only catalog.csv, the split and selected ZIPs. Avoid preparing the full collection on a small disk.

For a dry preparation, use the same command with prepare instead of run. No GPU or Harbor is needed to list or prepare. The generated task directory has standard instruction.md, task.toml, environment/, solution/ and tests/. The simulator builds from the pinned official NVIDIA Isaac Sim 5.1.0 container, and the agent builds from pinned public Python. The simulator download is large. On a proxied network pass Docker's standard build proxy arguments when building the two base images. Upstream robot files are checked by SHA256 and reused from the local asset cache.

~~~bash
python summarize.py --jobs work/agent-trial/jobs --manifest work/agent-trial/selection.csv
python verify_anonymous.py --data "$DATA" --output asset-readback.json
~~~

The first command reports expected, scored, missing and error counts. The second downloads every unique externally resolved asset without a Hub token, browser session or asset cache. Use verify_published.py after upload to check the complete HF dataset separately. Record the dataset revision, evaluation commit, hardware, adapter version and model settings with any reported score.
