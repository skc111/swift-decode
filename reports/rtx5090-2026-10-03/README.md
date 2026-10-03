# RTX 5090 单卡复现：首轮 gate 与 benchmark

2026-10-03 云端反馈：`gate-002` 通过，随后 `bench-001` 完成。
CUDA Graph 相比 eager 的解码速度约为 1.38 倍，且本轮输出一致。
正常性能路径下 MTP 的解码速度约为 graph 的 1.93–2.24 倍，但输出有分叉，
这组比值必须带上“输出不完全一致”的限定。

推理引擎、量化、Triton 算子、CUDA Graph 和 MTP 实现来自
[Token Rush](https://github.com/zyhector/token-rush)，本仓库基于上游 `b592bc6` 开始复现。
本次工作是实验入口、结果与环境记录、分叉诊断，以及 GDN 一致性模式的局部修正；
本页数值来自用户在云端运行的本次实验，未引用上游性能数字。

## 配置与证据

- GPU：单张 RTX 5090 32 GB；单请求、单流、batch size 1。
- 模型：`zyhector/Qwen3.8-27B-TokenRush-int4g128`，revision
  `29a49013c25005b32d436efc5324a4fcfd03bacd`；使用本地权重，离线执行。
- 环境：torch `2.14.0+cu130`、Triton `3.8.0`、Transformers `5.17.0`；
  FLA `516143e31fce09925e6c39ac37148444bad176c4`，来自运行前已反馈的环境检查。
- 本次运行前云端同步到 `study` 的 `4720103`。
- 后端 Triton，BF16 KV，MTP depth 3；三条输入（英文、代码、中文），greedy，关闭 thinking。
- 每次固定输出 128 token，忽略 EOS；每模式每输入预热 2 轮、测量 3 轮，合计 27 个测量请求。
- 最大上下文分配 4096，prefill chunk 512，draft vocabulary 上限 131072；模式运行顺序 eager、graph、mtp。

[bench-001.summary.json](bench-001.summary.json) 从用户粘贴的云端汇总提取，保留全部数值，
仅规范化 JSON 排版。它不是完整 `runs/` 备份，也没有原始逐请求 token、时间事件和环境明细。
`gate-002` 的通过依据是用户反馈的终端 `PASS`；未在本地伪造其 `summary.json`。
云端应继续保留 `runs/bootstrap/`、`gate-001/`、`diagnose-mtp-001/`、
`probe-mtp-head-001/`、`trace-verify-001/`、`gate-002/`、`bench-001/`。

本次 benchmark 汇总保存的指纹：

- configuration：`974d85134dfdec64ecc1b2c545b87cf32317879b73a1d618dafc2ac0a9493e62`
- runtime：`fd21c37ecb2eda5231e0f61427e6a1d42c2686c09194f2cd56de79dad1b947b6`

实验入口已在启动 benchmark 前校验其 gate 状态、配置和运行环境匹配。
这份报告位于 `reports/`，不进入实验源码指纹；实验代码未因整理报告发生变化。

## 性能结果

下表为每条输入 3 轮的中位数。解码速度单位为 token/s，定义为首 token 之后的
127 个实际输出 token 除以后续生成时间；比值是相应中位数之比。

| 输入 | eager | CUDA Graph | graph / eager | MTP† | MTP / graph† | draft 接受率 |
|---|---:|---:|---:|---:|---:|---:|
| 英文 | 70.76 | 97.34 | 1.38× | 188.24 | 1.93× | 49.67% |
| 代码 | 70.62 | 97.39 | 1.38× | 218.22 | 2.24× | 64.39% |
| 中文 | 70.57 | 97.38 | 1.38× | 188.21 | 1.93× | 49.67% |

† MTP 与 graph 的后续输出不完全相同，不能将这些比值写成等输出的无损加速。
接受率为 accepted drafts / proposed drafts，包含最后一次实际执行的完整投机步骤；
截断到 128 个输出后多算的尾部仍计入耗时，不计入解码速度的输出分子。

以下范围是三条输入各自中位数的最小值到最大值，不是单次运行的波动范围：

| 模式 | 首 token 时间 ms | 含首 token 的生成速度 token/s | 峰值 allocated GiB | 峰值 reserved GiB |
|---|---:|---:|---:|---:|
| eager | 366.7–370.1 | 59.00–59.22 | 16.127 | 16.533 |
| CUDA Graph | 369.1–370.0 | 76.46–76.49 | 16.127 | 16.537 |
| MTP | 374.4–375.2 | 121.96–133.73 | 17.113–17.114 | 17.701 |

首 token 时间包含状态重置、prefill 和相关草稿准备，不含加载、编译、图捕获、tokenizer 或网络。
显存来自 PyTorch allocator，包含常驻权重、缓存和图，不能当作整卡占用或纯 KV 用量。
MTP 相比 graph 的峰值 allocated 约增加 0.986 GiB，reserved 约增加 1.164 GiB。

eager 的代码输入有一轮降至 60.95 token/s（该组中位数 70.62），中文最低为 65.73
token/s（中位数 70.57）。不能仅凭三轮数据判断其原因，或把它们当作稳定的尾延迟统计。

## 输出一致性与 GDN 调整

`gate-001` 最初失败。逐步诊断发现，对相同的 prefill 状态和同一组输入，四行验证
使用完整 GDN value head（`BV=128`）时，第一个 block 的 1365 处观测和全部 logits
均与逐 token 计算一致；仅统一量化投影的调优配置没有消除差异。
详细证据见 [MTP_DIAGNOSTICS.md](../../experiments/MTP_DIAGNOSTICS.md)。

提交 `4720103` 将 GDN 的完整 value head 设置接入 `consistent=True`。
用户随后反馈 `gate-002` 通过：在这个模式、这些输入和重复轮次上，三种路径逐 token 一致。
这是 int4 checkpoint 内部路径对照，没有执行独立的 Hugging Face / BF16 模型数值验证。

`bench-001` 按现有协议使用 `consistent=False`：单 token 采用正常 GEMV，
GDN 恢复上游的按行数选择分块。它与一致性 gate 是两套数值路径，输出比较结果为：

- eager 与 graph：三条输入、各三轮全部一致。
- MTP 与 graph：英文在输出索引 20、代码在 61、中文在 69 首次分叉，均从 0 开始，
  每条输入的三轮都是同一个首次分叉位置。仅凭这些位置不能证明分叉后的 MTP 三轮全文相同。
- `all_equal=false`，共 9 项 MTP 对比失败；benchmark 的 `status=complete` 表示测量完成，
  不表示输出一致性通过。终端的 `WARNING` 已保留在本报告的结论中。

## 本轮结论与下一步

第一阶段的环境准备、模型加载、完整模型路径 gate 和首轮性能基线已经跑通。
可以报告复现过程、CUDA Graph 在本轮等输出条件下的收益，以及带输出差异限定的 MTP 速度。
不能将上游引擎或 MTP 算法写成本人的原创实现，也不能宣称已验证通用生成质量或服务吞吐。

下一步先从这张表解释 eager、CUDA Graph 和 MTP 分别改变了哪一段执行路径。
后续若要给出 MTP 的等输出性能比，应补充启用一致性路径的计时对照并重新验证匹配 gate，
不能把本次正常路径的速度直接套到已通过一致性 gate 的路径上。
