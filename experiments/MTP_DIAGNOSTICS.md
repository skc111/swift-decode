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

本地 CPU 测试只覆盖结果定位、验证行对齐和记录保护；诊断的 GPU 路径仍需云端执行。
即使在同一引擎中输出相同，也必须重新通过各模式独立进程的正式 gate，才能运行匹配的 bench。
