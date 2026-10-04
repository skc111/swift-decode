# 环境与小规模检查

这里保留上游的设备诊断脚本，供环境变化或定位问题时使用。
Swift Decode 当前云端环境已完成版本、Triton、CUDA Graph 和 FLA 小规模检查；
完整模型结果见 [实测报告](../../reports/rtx5090-2026-10-03/bench-002.md)。

| 文件 | 检查内容 |
|---|---|
| [check_stack.py](check_stack.py) | Triton、CUDA Graph、FLA/GDN 小张量数值与执行检查 |
| [check_env.sh](check_env.sh) | GPU、驱动、设备环境及 profiler 可用性 |
| [check_bandwidth.py](check_bandwidth.py) | 在当前设备上测读取带宽 |
| [check_gemv_sol.py](check_gemv_sol.py) | 模型相关矩阵形状的 GEMV 执行效率 |

现有 [环境准备说明](../../experiments/README.md) 会导入 `check_stack.py` 的函数，
因此这些检查继续保留原路径。带宽和 GEMV 检查的存在不代表当前云端已运行并归档它们。

当前环境的版本与 FLA 提交核对入口是 `experiments.single_gpu --check-env`，
推理复现流程见 [一致性实验步骤](../../experiments/CONSISTENT_BENCH.md)。
旧机器安装方式可从 [上游历史索引](../../docs/UPSTREAM.md) 查阅；本目录不再提供旧机器的安装命令。
