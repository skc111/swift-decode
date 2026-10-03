# 输出一致的性能对照

首轮 `gate-002` 已通过，`bench-001` 使用正常性能路径并完成计时，但 MTP 与 graph 的
输出有分叉。历史数据与限定见 [首轮报告](../reports/rtx5090-2026-10-03/README.md)。
本轮要测量使用一致性路径时的 eager / CUDA Graph / MTP 性能，并实际检查输出。
入口已实现，新的 GPU gate 与 benchmark 结果待云端反馈。

## 配置与判定

新增 `--bench-kernels normal|consistent`，默认 `normal` 保持首轮 benchmark 的行为。
该参数在 gate 阶段表示计划配对的 benchmark 类型，两阶段必须使用同一取值。
所有 gate 仍使用一致性模式；`--bench-kernels consistent` 使计时阶段也使用：

- `Engine(consistent=True)`：单 token 使用多行量化算子路径，GDN 固定完整 value head。
- 与 gate 一样的 recurrent slot 数量，包括 eager 和 graph。这样短 prefill 选择 fused
  路径的阈值也与 gate 相同，状态分配差异不会混入这组对照。
- 相同模型、输入、生成参数、源码和运行环境；配置指纹包含计划使用的 benchmark 类型。

一致性 benchmark 完成后检查两件事：

1. 本次所有模式、所有测量轮次的输入 token 和完整输出是否与本轮 graph 第 0 轮一致。
2. 同一批结果是否还与对应 gate 保存的 graph 第 0 轮输入和输出一致。

任一检查失败，`summary.json` 和 `suite.json` 都标记 `failed`，进程返回非零状态，
计时数据作为诊断证据保留。不能把失败运行的时间直接作为等输出性能结果。
两项都通过时，benchmark 状态为 `complete`，终端明确打印通过消息。
这些检查仅证明本次工作负载的逐 token 一致性，不是独立 HF / BF16 模型精度验证。

## 云端执行

同步新代码后，先运行第 1 步并查看结果。入口代码变化后，旧 `gate-002` 不再匹配当前
源码指纹，需要新的 `gate-003`；保留所有历史目录。

```bash
cd /root/shared-nvme/swift-decode &&
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/venvs/swift-decode/bin/python -u -m experiments.single_gpu \
  --stage gate --bench-kernels consistent \
  --model /root/shared-nvme/models/Qwen3.8-27B-TokenRush-int4g128 \
  --backend triton --kv bf16 --mtp-depth 3 \
  --output runs/gate-003
```

第 1 步通过后，再运行第 2 步。保持同一张 GPU 没有其他任务，使用同一版本代码。

```bash
cd /root/shared-nvme/swift-decode &&
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/venvs/swift-decode/bin/python -u -m experiments.single_gpu \
  --stage bench --bench-kernels consistent \
  --model /root/shared-nvme/models/Qwen3.8-27B-TokenRush-int4g128 \
  --backend triton --kv bf16 --mtp-depth 3 \
  --gate-dir runs/gate-003 --output runs/bench-002 &&
cat runs/bench-002/summary.json
```

沿用三条输入、128 个输出 token、2 轮预热和 3 轮测量。不会下载额外模型、安装依赖或
改变系统环境。本轮先保持原有计时规则，数据回来后再与历史报告分别列示。

## 新增记录

- `configuration.json`：protocol 2，包含 `bench_kernels`；不匹配的配对会在启动模型前被拒绝。
- `summary.json`：`kernel_mode` 表示实际数值路径，`output_comparison` 是本轮内部对照，
  `gate_output_comparison` 是与 gate 的逐 token 对照。正常性能模式不做后一项，记录为 null。
- `gate_reference.json`：从 gate 只读取得的输入／输出参考副本；不会修改原 gate 文件。
- 每条 worker JSONL 记录：`execution_options` 保存传给引擎的 `consistent` 和 `max_spec`。
- `comparison.csv`：新增实际 `kernel_mode`、本轮整体 `outputs_equal` 和
  `gate_outputs_equal`。两列一致性状态作用于整组运行；正常路径的 gate 对照列为空。

一致性路径的吞吐与正常路径应分别报告。不能拿正常路径的最快速度配上一致性路径的
正确性结论，也不能将路径和状态分配同时变化的性能差值全部归因于 GDN 一个设置。
推理引擎、算子和 MTP 算法来自 Token Rush；本次新增的是显式实验配置和结果核验。
