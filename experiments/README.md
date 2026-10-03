# 单卡复现：先跑通，再比较

这一版只做一件事：在自己的 RTX 5090 上，用相同的模型和输入比较
eager、CUDA Graph、MTP。DFlash 留作后续可选项，不做双卡，也不先承诺加速比例。

推理引擎、量化、算子、MTP 和 DFlash 实现来自
[Token Rush](https://github.com/zyhector/token-rush)，起点为 `b592bc6`。
本次新增的是 `experiments/` 评测入口、CPU 测试，以及一个显式选择量化后端的开关。
**目前只通过了本地 CPU 测试，尚未在租用的 5090 上验证。** 原仓库 `results/` 的数字不是本次实验结果。

## 1. 本地检查与代码同步

在仓库根目录执行，不需要安装 PyTorch：

```bash
python3 -m unittest discover -s experiments/tests -v
python3 -m experiments.single_gpu --plan --model /path/to/local-packed-model
```

`--plan` 只显示任务数，不下载模型、不使用 GPU，也不创建结果目录。

本地提交并推送 `study` 分支后，云端才能拿到新增文件。不要只拉 `main`：

```bash
# 本地；先确认 git diff 中没有不想提交的改动
git add .gitignore README.md experiments tokenrush/backend.py tokenrush/quant.py
git commit -m "Add single-GPU reproduction harness and CPU tests"
git push -u origin study

# 云端；仅第一次克隆时执行
cd /root/shared-nvme
git clone --branch study https://github.com/skc111/swift-decode.git
cd swift-decode
```

已有同名目录时不要覆盖它；进入目录，确认当前分支和本地改动，再 `git pull --ff-only`。

## 2. 先验证新环境，不覆盖镜像自带的 PyTorch

云端镜像提供的是 PyTorch 2.7.0a0 / CUDA 12.8；当前仓库的 `uv.lock` 则固定为
PyTorch 2.14.0+cu130、Triton 3.8.0、Transformers 5.17.0 和指定提交的 FLA。
这不是同一套环境。先独立安装并验证，不直接升级系统 Python，也不套用旧版 NCCL 的复制补丁。

以下命令在云端仓库根目录运行。`uv` 不存在时，先用单独的小环境安装它：

```bash
python3 -m venv /root/venvs/swift-tools
/root/venvs/swift-tools/bin/python -m pip install uv
export PATH="/root/venvs/swift-tools/bin:$PATH"
```

安装项目环境：

```bash
set -euo pipefail
export UV_PROJECT_ENVIRONMENT=/root/venvs/swift-decode
export HF_HOME=/root/shared-nvme/.cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
export PY=/root/venvs/swift-decode/bin/python
mkdir -p runs/bootstrap
df -h /root /tmp /root/shared-nvme
uv sync --locked --python /usr/bin/python3 --no-cache 2>&1 | tee runs/bootstrap/install.log
"$PY" -m experiments.single_gpu --check-env | tee runs/bootstrap/environment.json
```

环境放在系统盘，模型放在数据盘，避免 50 GB 数据盘同时塞进两份安装缓存和模型。
安装时仍需关注两个盘的剩余空间；上面不安装约 55 GB 的 BF16 权重。
`--locked` 的作用是锁文件不匹配时直接报错，而非悄悄更新依赖。
[uv 官方说明](https://docs.astral.sh/uv/concepts/projects/sync/)

`--check-env` 核对版本、FLA 提交、GPU 架构，并做一个 CUDA 矩阵乘法。
它通过后，再复用上游的小检查确认 Triton 和 CUDA Graph 能运行，无需先下载模型：

```bash
"$PY" -u - <<'PY' 2>&1 | tee runs/bootstrap/kernels.log
from scripts.env_check.check_stack import check_triton, check_cuda_graph
check_triton()
check_cuda_graph()
PY
```

任一步失败先停下，保留日志。尤其不能把 `torch.cuda.is_available() == True` 当作整个项目兼容。
当前先用 Triton 后端，不触发 Marlin 的 nvcc 编译；这不能保证 Triton 或 FLA 一定兼容驱动，仍以实际检查为准。
不要照搬上游另一台机器的 CUDA 13.3 安装命令或 forward-compat 库配置。

## 3. 下载上游指定的打包模型

这不是普通 Qwen 权重加载器。需要上游指定的 int4 打包目录，至少包含
`config.json`、`tokenrush.json`、`model-*.safetensors` 和 tokenizer 文件。
旧项目里的 Qwen3-8B / Qwen3-0.6B 不能替代它。

**环境检查通过后再执行这段下载，约 17 GB 是上游给出的大小，不是本机实测。**
先确认数据盘留有余量，且目标目录为空或确实是同一个模型：

```bash
set -euo pipefail
export MODEL=/root/shared-nvme/models/Qwen3.8-27B-TokenRush-int4g128
export MODEL_REPO=zyhector/Qwen3.8-27B-TokenRush-int4g128
export MODEL_REVISION=$("$PY" -c 'import os; from huggingface_hub import HfApi; print(HfApi().model_info(os.environ["MODEL_REPO"]).sha)')
test -n "$MODEL_REVISION"
printf 'repo=%s\nrevision=%s\n' "$MODEL_REPO" "$MODEL_REVISION" | tee runs/bootstrap/model-source.txt
/root/venvs/swift-decode/bin/hf download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$MODEL"
printf '%s\n' "$MODEL_REVISION" > "$MODEL/REVISION.txt"
```

按完整提交号下载是为了保留模型版本，而不是每次重新取 `main`。
[Hugging Face 下载说明](https://huggingface.co/docs/huggingface_hub/guides/download)
若 Hugging Face 不通，保留具体报错再处理网络或文件转移，不要先换成名字相近的模型。
目前还没有在这台服务器验证模型仓库的可达性。

## 4. 先检查输出，再测速度

仍在云端仓库根目录执行。每种模式单独起一个进程，避免上一次加载的模型和图驻留在下一次的工作进程中。
默认三个短输入：英文、代码、中文；固定输出 128 token，预热两轮、测量三轮，greedy，关闭 thinking。

```bash
# 用于检查输出一致性；不是性能结果
"$PY" -u -m experiments.single_gpu --stage gate --model "$MODEL" --output runs/gate-001

# 只有上一步通过才运行；不要并行跑别的 GPU 任务
"$PY" -u -m experiments.single_gpu --stage bench --model "$MODEL" \
  --gate-dir runs/gate-001 --output runs/bench-001
```

已有结果目录不会被覆盖，重跑请换名称。每个模式默认超时 1800 秒，超时会停止该进程组、记录失败，不继续后面的模式。
冷启动包括加载、编译和图捕获，不能根据 128 token 的输出时间估计整套任务耗时。

两次运行的模型、输入、生成参数、源码和环境必须匹配。更换 backend、KV dtype、MTP 深度、输入、输出长度或依赖后，需要重新跑 gate。
`gate` 使用上游的 `consistent=True`，使单 token 与多 token 验证尽量走一致的数值路径；
它逐 token 比较各路径和各重复轮次的完整输出，**不是独立的 Hugging Face / BF16 正确性验证**。
`bench` 关闭此选项，用正常路径计时，同时保存正常路径的输出差异。若不同，不能直接宣称“输出完全一致的无损加速”。

先跑完默认三条路径。之后再考虑：

- `--mtp-depth 1/2/3/4`：固定草稿长度，测接受率与速度的关系，不是上游自适应深度策略。
- `--kv fp8`：需新的 gate；仅三条短输入通过还不足以说明长上下文精度。
- `--backend marlin`：需先解决本机 CUDA 编译工具链兼容；显式选择后，编译失败直接报错。
- `--modes eager,graph,mtp,dflash --draft-model /local/draft`：额外需要上游指定的 DFlash2 模型，默认不会下载。

## 5. 怎么读结果

完整保留 `runs/gate-001/`、`runs/bench-001/` 和 `runs/bootstrap/`。

| 文件 | 内容 |
|---|---|
| `suite.json` | 完整命令、各进程退出码、是否真正完成 |
| `configuration.json`、`environment.json` | 参数、模型元数据、依赖、驱动、GPU、代码指纹 |
| `sources/`、`tracked_changes.patch` | 当时的实验和引擎源码，包括尚未提交的新文件 |
| `eager.jsonl`、`graph.jsonl`、`mtp.jsonl` | 每个请求的 token、文本、逐 step 事件、时间和显存数据 |
| `*.log` | 加载、编译和错误信息 |
| `summary.json`、`comparison.csv` | 输出对照、每个输入各轮指标的中位数和范围；CSV 仅 bench 生成 |

计时规则：

- `first_token_s`：已分词输入开始执行到首 token 可读，包含状态重置和草稿模型准备，不含加载、tokenizer、网络。
- `decode_output_tok_s`：首 token **之后**实际输出的 token 数 / 后续耗时。仅输出一个 token 时此值为空。
- `output_tok_s`：全部实际输出 token 数 / 首 token 加后续生成总时间。
- 投机最后一步可能多算，但只输出到指定长度；完整计算时间保留，并记录 `discarded_tail_tokens`，不拿多出的 token 抬高分子。
- 默认忽略 EOS，方便固定工作量；`--respect-eos` 可改为遇到 EOS 停止。固定长度实验不是自然生成质量评测。
- 记录真实 step 时间，不把一次投机返回的多个 token 虚构成等间隔到达。这里的 TPOT 是平均值，不提供服务端 ITL p95。
- 峰值显存来自工作进程的 PyTorch allocator，包含常驻权重/缓存/图，不是 `nvidia-smi` 整卡占用，也不等于 KV 用量。
- 模型记录包含元数据 SHA256 与权重大小/mtime，没有重新对十几 GB 权重计算内容哈希；保留下载来源和 revision。
- 模式按 `--modes` 的顺序运行；确认小幅收益时还需调换顺序复测，不能只凭同一进程的三轮就排除温度、频率和运行顺序影响。

这组数是**单请求、单流的引擎速度**。既不能与前一个项目的 256 请求总吞吐直接相比，
也不能据此写“比生产 vLLM 快多少”。下一步是否扩展，先看这次实测里哪一条路径真正有收益。
