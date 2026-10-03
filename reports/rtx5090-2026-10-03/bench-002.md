# RTX 5090：输出一致的性能对照

2026-10-03，用户反馈 `gate-003` 通过，随后 `bench-002` 完成。
本轮使用 `--bench-kernels consistent`：eager、CUDA Graph、MTP 的测量输出逐 token
一致，且全部与保存的 gate 参考一致。MTP 的解码速度为 CUDA Graph 的 **2.00–2.26 倍**。
第一版单卡复现已完成环境验证、分叉诊断、路径一致性 gate 和匹配配置的性能对照。

引擎、量化权重、Triton 算子、CUDA Graph 与 MTP 算法来自
[Token Rush](https://github.com/zyhector/token-rush)，上游起点为 `b592bc6`。
本仓库新增实验入口与记录、诊断工具、GDN 一致性模式的局部修正，以及配对结果核验。
这里的速度来自本次云端实验，与上游 README 中的性能数字分开记录。

## 配置和证据

- 单张 RTX 5090 32 GB，单请求、单流、batch size 1。
- 模型：`zyhector/Qwen3.8-27B-TokenRush-int4g128`，revision
  `29a49013c25005b32d436efc5324a4fcfd03bacd`，使用本地文件离线运行。
- 环境沿用已验证的 torch `2.14.0+cu130`、Triton `3.8.0`、Transformers `5.17.0`，
  FLA `516143e31fce09925e6c39ac37148444bad176c4`。
- Triton 后端、BF16 KV、MTP depth 3；greedy、关闭 thinking，忽略 EOS，固定输出 128 token。
- 英文、代码、中文三条输入；每模式每输入预热 2 轮、测量 3 轮，共 27 个测量请求。
- 最大上下文分配 4096，prefill chunk 512，draft vocabulary 上限 131072。
- 三种模式均使用 `consistent=True, max_spec=3`，即相同的 target recurrent slot 数量。
  单 token 使用多行量化算子路径，GDN 固定完整 value head；模式依次在独立进程中运行。
- 使用提交 `2244506` 新增的 protocol 2 实验入口；命令见
  [一致性 benchmark 说明](../../experiments/CONSISTENT_BENCH.md)。

[bench-002.summary.json](bench-002.summary.json) 提取自用户粘贴的云端终端输出，保留 JSON
内的原始数值，仅统一换行。已在本地核对两组各 27 条检查覆盖全部模式、输入和轮次，
以及 9 组性能汇总各有 3 轮测量；没有在本地重放 GPU 任务。

`gate-003` 的通过依据是用户此前反馈的终端 PASS；本地未构造其原始汇总。
收到的 benchmark summary 不含 Git HEAD 或逐请求 token。实际运行的提交、源码快照、
环境、完整 token 和 step 事件以云端 `runs/gate-003/`、`runs/bench-002/` 为准，
应与前几轮 `runs/` 一起保留。本报告及归档汇总不是完整运行目录备份。

汇总中的指纹：

- configuration：`3d7fd1c520deb412a7e396448f23f4b479c1f08bf99f978e2299a32e2b4fb8bc`
- runtime：`fd21c37ecb2eda5231e0f61427e6a1d42c2686c09194f2cd56de79dad1b947b6`

本轮 runtime 指纹与 `bench-001` 一致。配置指纹变化包含实验源码和 benchmark 路径变化；
运行环境指纹相同不表示 GPU 频率、温度或 CPU 负载始终相同。

## 输出检查

| 检查 | 参考 | 结果 |
|---|---|---|
| 本轮跨模式及跨轮次 | 本轮同一输入的 graph 第 0 轮 | 27/27 通过 |
| benchmark 与 gate | `gate-003` 同一输入的 graph 第 0 轮 | 27/27 通过 |

两组检查均要求输入 token 和完整输出 token 相同，`first_difference` 均为 null。
汇总为 `status=complete`、`kernel_mode=consistent`，两个 `all_equal` 均为 true。
这两组检查使用同一批 27 个测量请求，不能记成 54 个独立样本。

结论限定于这三个短输入、128 token 和实际重复轮次的 int4 模型内部路径对照。
没有执行独立的 Hugging Face / BF16 模型正确性验证，也没有验证通用生成质量。

## 性能结果

下表是每组 3 轮测量的中位数，单位为 token/s；加速比是对应中位数之比。
解码速度为首 token 之后的 **127 个实际输出 token / 后续生成时间**。

| 输入 | eager | CUDA Graph | MTP | graph / eager | MTP / graph | draft 接受率 |
|---|---:|---:|---:|---:|---:|---:|
| 英文 | 66.37 | 88.88 | 177.54 | 1.34× | 2.00× | 48.72% |
| 代码 | 59.72 | 88.88 | 200.56 | 1.49× | 2.26× | 60.87% |
| 中文 | 53.67 | 88.81 | 188.30 | 1.65× | 2.12× | 53.06% |

接受率为 accepted drafts / proposed drafts。本实验每步提出 3 个草稿，因此每步平均
可用输出数为 `1 + 3 × 接受率`，三条输入约为 2.46、2.83、2.59；末步截断前计数。
这不等于加速比：每步仍需草稿生成、target 验证和状态处理。
末步超出 128 token 的计算保留在耗时中，多算的尾部不计入速度的输出分子。

下列范围是三条输入各自中位数的最小值至最大值，不是单次请求的波动范围。

| 模式 | 首 token 时间 ms | 含首 token 的生成速度 token/s | 峰值 allocated GiB | 峰值 reserved GiB |
|---|---:|---:|---:|---:|
| eager | 369.2–387.3 | 46.48–56.14 | 16.549 | 16.955 |
| CUDA Graph | 370.5–387.6 | 70.47–71.11 | 16.549 | 16.959 |
| MTP | 380.6–394.7 | 115.14–125.61 | 17.113–17.114 | 17.701 |

MTP 相比 graph 的峰值 allocated 约增加 0.564 GiB，reserved 约增加 0.742 GiB。
显存来自 PyTorch allocator，包含常驻权重、缓存和图；不是整卡占用或纯 KV 用量。
eager / graph 在这一组也分配了与 gate 一致的 target slots，因此不能把此增量
套到正常路径或其他配置上。

首 token 时间包含状态重置、prefill 和相应草稿准备。计时不含权重加载、编译、
图捕获、tokenizer 或网络；每个解码 step 包含实际取回 token 与同步的开销。
MTP 的约 2 倍收益描述的是后续解码，不能用作首 token 延迟或整个请求的加速倍数。

## 波动与结论范围

| 输入 | eager 三轮速度范围 | graph 三轮速度范围 | MTP 三轮速度范围 |
|---|---:|---:|---:|
| 英文 | 63.84–70.09 | 88.72–88.88 | 177.13–177.55 |
| 代码 | 50.92–61.71 | 88.82–88.90 | 200.35–200.71 |
| 中文 | 51.16–68.38 | 88.68–88.90 | 188.23–188.49 |

eager 波动较大，因此 graph / eager 的 1.34–1.65 倍只作为本轮观察，不能全部归因于
CUDA Graph 消除了多少 launch 开销。本轮 graph 和 MTP 的三轮速度更集中，
以 MTP / graph 的等输出对照作为主要性能结果。三个输入、各三轮仍不足以支持
稳定的尾延迟、跨运行置信区间或所有工作负载的性能结论。

[bench-001 首轮报告](README.md) 保留正常路径的原始数据：其 MTP 输出有分叉。
本轮一致性路径同时涉及单 token 量化路径、GDN 分块和 target slot 配置，
不能将两轮速度或显存差异全部归因于 GDN，也不能把历史最快速度与本轮一致性结论拼接。

可以对外表述：基于 Token Rush，在单张 RTX 5090 上完成 int4 模型推理复现、
路径一致性诊断与性能对照；在本次三条短输入、128 token 的等输出实验中，
MTP 相比 CUDA Graph 的解码速度约为 2.00–2.26 倍。
不据此宣称原创推理引擎、对 BF16 无精度损失、多请求服务吞吐或优于其他框架。

接下来从 [面试讲解笔记](INTERVIEW_NOTES.md) 理解这三条执行路径和诊断过程。
