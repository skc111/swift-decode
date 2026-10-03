# MTP gate 分叉诊断

2026-10-03 用户反馈的云端 `runs/gate-001`：Triton、BF16 KV、MTP depth 3，
eager 与 graph 的三条输入逐 token 相同，各模式三轮输出各自稳定。
MTP 在英文、代码、中文输入的输出索引 20、61、69 首次分叉（索引从 0 开始）。
此 gate 失败，尚无获准运行的 benchmark；这些是本次复现结果，不是上游性能数据。

`diagnose_mtp.py` 复用上游引擎和现有 Adapter，使用 gate 保存的输入 token 与配置。
它在一个加载了 MTP 的引擎中依次运行 graph 和 MTP，记录：

- 新输出是否重现原 gate 的 graph / MTP 输出；不同进程的自动调优结果可能不同。
- MTP 收集到的 accepted drafts / bonus token 是否逐行等于 target logits 的 argmax。
- 首次分叉的验证行、两条路径的 top logits、候选分数与差值。
- 已提交 token、位置和 recurrent slot 的计数，以及 Triton 多行算子的调优配置。

代码检查发现的候选因素：`consistent=True` 会打开 `ROWS_FOR_ONE`，但多行量化算子
仍按 `M` 分别调优；GDN 在 `M >= 4` 时会改变 `BV` 分块。
它们尚未被证明是本次分叉的原因。小的 logits 差值也不能单独证明状态提交或算子实现正确。

在云端仓库根目录、保持原实验环境，先同步 `study`，再运行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/venvs/swift-decode/bin/python -u -m experiments.diagnose_mtp \
  --gate-dir runs/gate-001 --prompt english --output runs/diagnose-mtp-001
```

原 gate 的源码、模型记录和环境须匹配；可以新增诊断文件，但不能先改掉原实验代码。
结果目录必须是新目录。诊断只读取原 gate，结果、环境和源码快照保存在独立目录。
logits 拷到 CPU 会同步 GPU，因此此入口没有性能含义。引擎与算子实现仍来自 Token Rush。

本地 CPU 测试只覆盖结果定位、验证行对齐和记录保护。
即使在同一引擎中输出相同，也必须重新通过各模式独立进程的正式 gate，才能运行匹配的 bench。

用户随后反馈 `runs/diagnose-mtp-001` 在英文输入上逐 token 重现原 gate 的两条输出，
collector 与位置／slot 计数检查均通过。输出索引 20、verify row 1 处：graph 的
` executed` / ` granted` logits 为 21.75 / 21.625，MTP target 则同为 21.625，
argmax 选择了较小 token ID 的 ` granted`。这解释了选词变化，但没有定位算子根因，
计数正确也不代表 KV 或 recurrent state 的数值内容已通过验证。

保存的调优配置中，`lm_head` 的 `(N=248320, K=5120)` 在 M=1 时使用
`BLOCK_N=64, BLOCK_K=256, num_stages=3`，M=4 时使用
`BLOCK_N=64, BLOCK_K=512, num_stages=2`，两者均为 4 warps。
尚不能据此断定该配置差异导致了本次分叉。

`probe_mtp_head.py` 恢复这些已记录的自动调优选择，只重放到首次分叉，取出实际
post-norm hidden。对每份冻结的 hidden 分别做单行、四行 head 投影，再只把四行
head 的配置暂时设为单行配置复测。它同时比较 raw / MTP hidden 是否已不同，
用于区分投影本身与更早计算的影响。临时缓存调整只在这个诊断进程中存在。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/venvs/swift-decode/bin/python -u -m experiments.probe_mtp_head \
  --diagnosis-dir runs/diagnose-mtp-001 --output runs/probe-mtp-head-001
```

用户随后反馈 `runs/probe-mtp-head-001` 完成并恢复了全部 15 条调优记录。
在这次英文分叉上，每份冻结 hidden 的单行、四行、统一配置投影的全部 logits
逐值相同；MTP hidden 重投影也完全重现捕获的 target logits。
但 raw / MTP hidden 的 5120 个值中有 4359 个不同，最大绝对差 0.125，平均绝对差
约 0.01319。证据将本次偏差定位到 head 之前；不能据此推广为所有输入的 head 等价性。

下一步使用 `trace_verify.py` 从相同的 target 状态和输入定位最早观察到差值的算子：

1. 比较 raw / MTP prefill 后的 hidden、KV 有效前缀、live recurrent state 和 conv ring。
   若这里已经不同，先报告 prefill 差异。
2. 按诊断保存的 MTP block 顺序，从 raw 的已提交前缀恢复同一份状态快照，分别逐 token
   和一次四行计算相同输入（committed token 与实际 drafts）。记录 norm、量化投影、
   GDN、attention 和 MLP 激活；split-K partials 按 token 轴对齐。
   batch 保留被拒绝的 draft 尾部，但只根据原步骤实际输出对应的行选择分叉位置。
3. 找到第一个存在差值的 block 后，分别测试 GDN 使用整个 value head（`BV=V`）、
   target 多行量化算子使用单行的调优配置，以及同时使用这两项设置。
   每组都恢复同一个初始状态，临时设置退出后恢复。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/venvs/swift-decode/bin/python -u -m experiments.trace_verify \
  --diagnosis-dir runs/diagnose-mtp-001 --output runs/trace-verify-001
```

终端打印最早差值和各控制组摘要，完整算子差值保存在新目录的 `trace.json`。
这是固定输入的 eager 算术诊断，会同步并复制激活到 CPU，不测性能。
它隔离同一初始状态下批量计算的影响；后续 block 使用 raw 前缀状态，不能替代对
MTP 实际提交后的全部状态内容的检查，也不能单凭一个算子的差值宣称根因已确认。
引擎、算子和 MTP 算法仍是上游实现，本仓库新增的是复现实验与诊断入口。

本地 CPU 测试覆盖输入／输出行对齐、缓存干预及追踪 hook 的恢复；算子数值仍需云端验证。
原 gate 仍失败，benchmark 仍未运行。
