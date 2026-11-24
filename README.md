# RoboUniverse-10K evaluation

<p align="center">
  <a href="https://arxiv.org/abs/2611.31002"><img alt="arXiv: 2611.31002" src="https://img.shields.io/badge/arXiv-2611.31002-b31b1b?style=flat-square"></a>
  <a href="https://lab.einsia.ai/robo-universe/"><img alt="Project Page: Website" src="https://img.shields.io/badge/Project%20Page-Website-29618a?style=flat-square"></a>
</p>

![Task taxonomy for 10,807 RoboUniverse-10K environments, with examples across eight Franka and AgileX task families.](assets/task-taxonomy.png)

*From the paper: task families and example environments across the full RoboUniverse-10K collection.*

Run a coding agent or a reference controller against one physical manipulation environment or the frozen 500-task selection. A Docker Compose simulation service owns the scene and evidence. The agent gets the task instruction and an observation/action client. Harbor runs the verifier separately after collection. A successful verdict earns 1, failure or inconclusive earns 0; an infrastructure exception is not a scored attempt.

![A coding agent interacts with Isaac Sim through the Harbor harness; a separate verifier scores the final state.](assets/harbor-evaluation.png)

*From the paper: a Harbor evaluation episode for the task of covering a pot with its lid.*

## Requirements

- **Model server.** Follow the [SGLang deployment recipe for Qwen3.6-35B-A3B (BF16)](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6) for GPU and runtime requirements. We used one B300 per server, SGLang 0.5.21 and PyTorch 2.13.0+cu132. Use the RoboUniverse launch command below to load its fine-tuned weights and bundled chat template.
- **Harbor host.** The tested configuration is Linux x86_64, an NVIDIA RTX PRO 5000 **72 GB Blackwell**, NVIDIA driver **580.105.08**, NVIDIA Container Toolkit 1.20.2, Docker 26.1.5 with Compose 2.40.3, Python 3.12 and Harbor 0.25.0. One simulation uses one RTX GPU. Its container memory limit varies from 18 to 44 GiB by task. Reserve additional host RAM and disk for Docker builds, assets, recordings and verifier output.
- **Access.** Review the [NVIDIA Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/NVIDIA_Omniverse_License_Agreement.html) before enabling the EULA flag. Setup needs HTTPS access to the public container registries, Python, npm, Debian, Hugging Face and upstream robot assets. The preinstall below uses registry.npmjs.org. Harbor’s native installer also uses downloads.claude.ai when the requested binary is absent. The simulation host and its agent containers must be able to reach the model endpoint.

## How to run

### 1. Serve the released model

On the **model server**, use the CUDA-compatible SGLang environment from the recipe above with `sglang==0.5.21`. Install the Hugging Face CLI in that environment if needed. Download the fixed revision, then start SGLang in a terminal that will stay open.

~~~bash
python -m pip install huggingface_hub
hf download Einsia/RoboUniverse-35B-A3B \
  --revision d787a4b4a26bb5ff868c6f14b8fd0b77288d98ba --local-dir ./RoboUniverse-35B-A3B

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python -m sglang.launch_server \
  --model-path ./RoboUniverse-35B-A3B --served-model-name RoboUniverse-35B-A3B \
  --host 127.0.0.1 --port 30000 --tp-size 1 --context-length 262144 \
  --tool-call-parser qwen3_coder --image-processor-backend pil \
  --moe-runner-backend triton --kv-cache-dtype bfloat16 --mem-fraction-static 0.85
~~~

Keep the bundled chat template and generation defaults. Leave the reasoning parser and reasoning effort unset. The bundled template defaults to thinking off.

### 2. Connect Harbor and build the images

On the **Harbor host**, install the evaluator. The examples assume the host can reach the model at `127.0.0.1:30000`. Set `ANTHROPIC_BASE_URL` to the same model as seen from inside a Docker container. The example below forwards the model port directly from the Harbor host. If both addresses already work, skip the SSH command and use your existing endpoints.

~~~bash
git clone https://github.com/Einsia/RoboUniverse-10K.git
cd RoboUniverse-10K
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt huggingface_hub==2.4.0
~~~

If you need forwarding, open a second terminal on the Harbor host, set `MODEL_SSH_HOST` to your model server’s SSH login and keep this command running.

~~~bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:30000:127.0.0.1:30000 \
  -L "$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'):30010:127.0.0.1:30000" \
  "${MODEL_SSH_HOST:?Set MODEL_SSH_HOST to the model server SSH login}"
~~~

Back in the Harbor shell, check the host endpoint and set the container endpoint.

~~~bash
BRIDGE_IP=$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')
export ANTHROPIC_BASE_URL="http://${BRIDGE_IP}:30010"

curl --noproxy '*' -fsS http://127.0.0.1:30000/v1/messages \
  -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"RoboUniverse-35B-A3B","max_tokens":16,"messages":[{"role":"user","content":"Reply OK"}]}'
~~~

The response should contain an assistant message from `RoboUniverse-35B-A3B`. Build both base images from the public Dockerfiles. Installing the pinned Claude Code binary during the image build avoids a network installation during each timed trial. If your network requires a proxy, pass Docker's standard build proxy arguments to all three builds.

~~~bash
docker build -t robouniverse-sim:5.1.0-v1 docker/sim-base
docker build -t robouniverse-agent:base docker/agent-base
docker build -t robouniverse-agent:1 - <<'DOCKER'
FROM robouniverse-agent:base
RUN set -eu; \
    curl -fsSL --retry 3 -o /tmp/claude.tgz https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/claude-code-linux-x64-2.1.283.tgz; \
    echo 'd14ec0fca400151092c926928fea7e8e38231f7a90e6e1f8eeb2a55c24676afb  /tmp/claude.tgz' | sha256sum -c -; \
    tar -xz -C /usr/local/bin --strip-components=1 -f /tmp/claude.tgz package/claude; \
    rm /tmp/claude.tgz; \
    claude --version | grep -Fx '2.1.283 (Claude Code)'
DOCKER
docker run --rm robouniverse-agent:1 curl --noproxy '*' -fsS "$ANTHROPIC_BASE_URL/v1/models"
docker run --rm -e ANTHROPIC_BASE_URL -e ANTHROPIC_API_KEY=local-test \
  -e CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 -e MAX_THINKING_TOKENS=0 \
  robouniverse-agent:1 claude --bare --strict-mcp-config --tools "" \
  --model RoboUniverse-35B-A3B --no-session-persistence -p "Reply with the single word READY."
~~~

The second check sends a message from the pinned Claude Code client to the model through the same container endpoint. With the SGLang command above, the API key is a local placeholder. The tested client printed a model-catalog warning followed by `READY`.

### 3. Run one held-out task

Keep the same Harbor shell. If you already have the dataset, set `DATA` to the absolute path containing `catalog.csv`, `splits/` and `tasks/`. The commands below download missing files and check the local catalog and split against the pinned release. The catalog gives the SHA-256 and ZIP path for each of the 10,807 environments. Each invocation of `prepare` or `run` creates a fresh work directory. The evaluator verifies the ZIP and upstream assets, then generates the agent’s connection file and the simulator’s matching instance file. Compose mounts each file read-only from that task’s directory. After upgrading from an earlier evaluator, prepare or run the task again from its ZIP.

~~~bash
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_DATASET=Einsia/RoboUniverse-10K
export HF_REVISION=ae170e8093dd6061765c14f2fb6a209432517d85
export DATA="${DATA:-$PWD/dataset}"
download_data() {
  hf download "$HF_DATASET" "$@" --repo-type dataset --revision "$HF_REVISION" --local-dir "$DATA"
}
for file in catalog.csv splits/heldout500.csv; do
  if [ ! -f "$DATA/$file" ]; then download_data "$file"; fi
done
(cd "$DATA" && sha256sum -c - <<'SHA256'
62ab803a5469e530e8670173829f2b6062101ae71a57a306994edc699a0bc638  catalog.csv
e555968421293744ce214ff0a993d60dd3b4484d36517290b74198551c6cdc11  splits/heldout500.csv
SHA256
) || exit 1
python evaluate.py list --data "$DATA" --split heldout500 | wc -l  # 500

CASE=000247_rotate_twist
archive=$(awk -F, -v id="$CASE" '$1 == id {print $3; exit}' "$DATA/catalog.csv")
if [ ! -f "$DATA/$archive" ]; then download_data "$archive"; fi

# Run the reference controller after reviewing the NVIDIA license.
WORK=$(mktemp -d "$PWD/smoke.XXXXXX")
python evaluate.py run --data "$DATA" --case "$CASE" --work "$WORK/oracle" \
  --agent oracle --gpu 0 --accept-isaac-eula
python summarize.py --jobs "$WORK/oracle/jobs" --manifest "$WORK/oracle/selection.csv"
~~~

The reference-controller run should report one scored success. With evaluator commit `9d96bc4`, we also ran `000247_rotate_twist`, `001332_press_switch` and `001734_press_switch` concurrently on two RTX GPUs. In each agent container, `roboenv describe` returned its own case ID, and all three reference-controller trials scored 1.

The model trial we verified used Harbor 0.25.0, the Claude Code adapter at version 2.1.283, model `anthropic/RoboUniverse-35B-A3B`, and `max_thinking_tokens=0`. On `000360_push_topple` the verifier awarded 1 after 225 physics steps. Regrading the saved evidence with the published task's `tests/grade.py` reproduced the original result.

### 4. Prepare the Final 500 and score a run

Download the ZIPs named by the split. The 500 ZIPs total 764 MiB. On the tested Linux filesystem, `prepare` expanded them into 39.5 GiB of task directories before Docker builds or trial logs. All 200 Franka and 300 AgileX tasks passed preparation. An independent anonymous download checked the SHA-256 of all 146 distinct upstream assets. Preparation can run without a GPU.

~~~bash
WORK=$(mktemp -d "$PWD/final500.XXXXXX")
python - "$DATA" > "$WORK/archives.txt" <<'PYTHON'
import csv
import sys
from pathlib import Path

data = Path(sys.argv[1])
with (data / 'catalog.csv').open() as f:
    archives = {row['case']: row['archive'] for row in csv.DictReader(f)}
with (data / 'splits/heldout500.csv').open() as f:
    cases = [row['case'] for row in csv.DictReader(f)]
assert len(set(cases)) == len(cases) == 500
for case in cases:
    if not (data / archives[case]).is_file():
        print(archives[case])
PYTHON
xargs -r -n 50 sh -c 'hf download "$HF_DATASET" "$@" --repo-type dataset \
  --revision "$HF_REVISION" --local-dir "$DATA" || exit 255' sh < "$WORK/archives.txt"
python evaluate.py prepare --data "$DATA" --split heldout500 --work "$WORK/prepared"
~~~

To run the reference controller directly from the downloaded ZIPs, use the `run` entry point with a fresh work directory. It runs one trial per case sequentially and invokes the separate verifier after collection. Keep the work directory private because it contains per-trial connection tokens and captures.

~~~bash
python evaluate.py run --data "$DATA" --split heldout500 --work "$WORK/oracle" \
  --agent oracle --gpu 0 --accept-isaac-eula
python summarize.py --jobs "$WORK/oracle/jobs" --manifest "$DATA/splits/heldout500.csv"
~~~

Report successful cases out of **500**, together with scored, missing and error counts. The summary's `success_rate_among_scored` uses only scored cases. A 20% result on the full split requires 100 distinct successful cases. A replay of selected historical successes has its own denominator. Record the evaluator commit, dataset and model revisions, hardware and inference settings with each result.

Allow the simulator collection hook its declared 1,800 seconds after the agent finishes, including its 1,770-second HTTP read timeout. Evidence export can take up to 1,650 seconds. For unscored trials, inspect the trial log and the collected `rollout/infra_error.json` when present. `python verify_anonymous.py --data "$DATA" --output asset-readback.json` checks upstream asset access across a complete local snapshot.

## Citation

If you use RoboUniverse-10K in your research, please cite:

~~~bibtex
@misc{liu2026robouniverse10kscalingeraagentic,
      title={RoboUniverse-10K: Toward the Scaling Era of Agentic Robotics via Real-to-Sim
              Environments},
      author={Jinkun Liu and Yingze Wang and Situ Wang and Deyao Hong and Mingju Gao and Wenyi
              Li and Yizhe Chi and Youjie Zheng and Kaisen Yang and Xiangyang Wang and
              Bingxiang He and Yuxin Zuo and Boyuan Chen and Yue Chen and Cheng Qian and
              Tianxing Chen and Huichi Zhou and Huaxiu Yao and Hao Tang and Hao Zhou and
              Ya-Qin Zhang and Calvin Xiao and Tiga Altman and Huan-ang Gao},
      year={2026},
      eprint={2611.31002},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2611.31002},
}
~~~
