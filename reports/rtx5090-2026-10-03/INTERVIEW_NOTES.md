# Swift Decode：从实测讲清楚项目

第一版实验已经跑通。先掌握下面的项目边界、三条执行路径和关键诊断，
再沿代码理解细节。结果来源是 [bench-002 报告](bench-002.md)；
此处讨论单请求、单流的推理执行，尚未进行多请求服务实验。

完整讲解见 **[Swift Decode 项目详解与面试准备](../../docs/SWIFT_DECODE_GUIDE_ZH.md)**：
包含模型与状态、CUDA Graph、MTP 接受与提交、诊断证据、计时口径、代码阅读顺序和
40 个面试问题。本文保留为快速复习版。

## 一分钟介绍

> 我基于开源 Token Rush，在单张 RTX 5090 上复现了一个 27B int4 模型的推理，
> 对 eager、CUDA Graph 和 MTP 做统一输入、生成参数和计时规则的实验。
> 我的工作集中在实验框架、环境与源码记录、正确性诊断和局部一致性修正。
> 最初 MTP 与普通解码输出不同，我通过重放和固定状态的算子对照，将问题定位到
> GDN 分块相关的数值路径差异，并补全一致性模式。
> 随后重新跑 gate，让性能测试也使用相同配置并核对完整输出。
> 在这三个短输入、128 token 的实验中，MTP 相对 CUDA Graph 的解码速度达到
> 2.00–2.26 倍，所有测量输出均与 gate 一致。

引擎、量化方案、Triton 算子、CUDA Graph 和 MTP 算法属于上游贡献。
本仓库完成的是复现、对照方法、诊断证据和一致性模式的局部修正。
不要把这个介绍扩大为独立开发推理引擎或证明量化相对 BF16 无精度损失。

## 三种模式分别做了什么

| 模式 | 本仓库实际调用 | 每步新输出 | 主要比较点 |
|---|---|---|---|
| eager | `engine.decode()`，然后 greedy 选词 | 1 个 token | 基础逐步执行路径 |
| CUDA Graph | `engine.step()` 重放提前捕获的 decode graph | 1 个 token | 减少重复的 CPU 调度和算子提交工作 |
| MTP | `engine.spec_step(3)` 重放草稿生成、验证与状态处理的 graph | 1–4 个 token | 一次 target 验证产生多个可用输出，摊薄每 token 的成本 |

调用入口在 [Adapter.step](../../experiments/gpu.py)，图捕获和重放在
[Engine](../../tokenrush/model.py) 的 `capture`、`step`、`capture_spec`、`spec_step`。
eager 模式仍使用上游优化过的量化与融合算子；“eager”描述调度方式，不能当作未经优化的
PyTorch 或 Hugging Face 基线。图捕获在计时之前完成，本轮没有衡量其冷启动成本。

MTP depth 3 表示草稿头连续猜 3 个后续 token，再由 target 验证。
例如草稿猜 `A B C`，target 在相同前缀下依次认可 `A B`，第三个位置选择 `D`，
这一轮就输出 `A B D`：接受前两个草稿，再输出 target 的纠正 token。
三个草稿全被接受时，还可输出一个 bonus token。已输出的 committed input 不重复计数。
逻辑对应 `Engine._spec_step`、`Engine._verify_step` 和 `Adapter.step`。

因此，草稿接受率不是加速倍数。代码输入接受率为 60.87%，固定深度 3 时每步平均
产生约 `1 + 3 × 0.6087 = 2.83` 个可用 token（末步截断前），实测解码速度比为 2.26。
草稿生成、验证和状态处理仍有成本，不能用接受率直接推算时间。

## 为什么需要那次诊断

首次 gate 中，eager 与 graph 一致，MTP 在英文第 20、代码第 61、中文第 69 个输出
索引开始不同，均从 0 计数。结果稳定重复，因此先沿真实执行路径定位差异。

1. 重放原结果，核对 emitted token 与 target argmax 一致，检查位置和 slot 计数。
   英文分叉处两个候选非常接近：graph 为 21.75 / 21.625，MTP 为 21.625 / 21.625。
   这解释了 argmax 变化，但计数正确还不能证明状态内容正确。
2. 冻结同一份 hidden，比较单行和多行 `lm_head`，logits 完全相同；两条路径的 hidden
   在进入 head 前已经不同。此证据将该次分叉的调查重点移到模型内部。
3. 从相同 prefill 状态和相同输入开始，对比逐 token 与四行验证的中间结果。
   第一个 block 的最早观测差异出现在 GDN 输出。单独统一 GDN 为完整 value head 后，
   该 block 的 1365 处观测全部相同；单独统一量化算子调优配置未消除差异。
4. 将完整 GDN head 接入 `consistent=True`，重新跑完整生成 gate，再运行相同路径
   的 benchmark。最终三种模式和保存的 gate 参考逐 token 相同。

详细原始证据见 [诊断记录](../../experiments/MTP_DIAGNOSTICS.md)。修正的是一致性模式
中的分块选择，没有重写上游 kernel 公式或放宽输出判定。浮点路径变化对相近候选可能
产生影响；这里的结论来自控制实验和后续 gate，不能只凭“小差值”就忽略失败。

## 怎么解释数字

主结果是同一路径、同一工作负载下的 **MTP / graph = 2.00–2.26**。
速度单位为首 token 之后的实际输出 token/s，不含加载、编译和图捕获。
首 token 时间单独记录，约为数百毫秒，不能声称也缩短为一半。

eager 的三轮速度波动较大，graph 和 MTP 在本轮更集中；因此不把较高的 MTP / eager
比值作为主要结论。先描述观测，再解释机制，不根据这些汇总反推 CPU、频率或温度原因。

MTP 的峰值 allocated 约 17.11 GiB，比本轮 graph 多约 0.56 GiB。
它包含常驻权重、状态、缓存和图等分配，不是整卡占用，也不能把差额全算成 KV。
本轮为匹配 gate，三种模式使用相同 target slot 数量；正常路径另有一份历史报告。

本次验证覆盖三条短输入和三轮重复，没有独立 HF / BF16 精度对照、长上下文实验或
多请求服务吞吐。进一步实验应由明确问题驱动；当前结果已经足够完成第一版复现展示。
