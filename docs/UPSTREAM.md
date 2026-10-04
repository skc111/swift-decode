# 上游来源与历史材料

Swift Decode 从 [Token Rush](https://github.com/zyhector/token-rush) 的
[`b592bc6`](https://github.com/zyhector/token-rush/tree/b592bc6d91f0ea925b6a0179cbe65d1362b96381)
开始复现。底层引擎、量化方案、算子、CUDA Graph 与 MTP 实现来自上游；本仓库新增
单卡实验框架、分叉诊断、GDN 一致性模式的局部修正及结果核验。

当前目录聚焦可运行代码、Swift Decode 自己的实验记录和项目讲解。上游的历史日志、
跨框架跑分脚本、图表、开发流水账和旧机器配置已从工作目录移除，原文件仍在 Git 历史中。
本次整理前的完整版本为
[`c185260`](https://github.com/skc111/swift-decode/tree/c185260943757c4b283147ee15cb3a948cc7b04d)。
这些旧数据是原作者的结果，不是 Swift Decode 的实测。

## 查阅上游历史

引擎源文件注释中使用的历史文档路径，也可按此表查阅固定版本。

| 历史路径 | 用途 | 固定版本 |
|---|---|---|
| `README.md`、`CLAUDE.md` | 原项目介绍、开发目标和旧机器说明 | [原始首页](https://github.com/zyhector/token-rush/blob/b592bc6d91f0ea925b6a0179cbe65d1362b96381/README.md)、[开发记录](https://github.com/zyhector/token-rush/blob/b592bc6d91f0ea925b6a0179cbe65d1362b96381/CLAUDE.md) |
| `results/` | 历次环境、基线、质量和性能日志 | [原始数据](https://github.com/zyhector/token-rush/tree/b592bc6d91f0ea925b6a0179cbe65d1362b96381/results) |
| `docs/progress.md` | 算子和推理路径的开发过程 | [开发过程](https://github.com/zyhector/token-rush/blob/b592bc6d91f0ea925b6a0179cbe65d1362b96381/docs/progress.md) |
| `docs/quantization.md` | GPTQ 校准、量化质量和语料来源 | [量化记录](https://github.com/zyhector/token-rush/blob/b592bc6d91f0ea925b6a0179cbe65d1362b96381/docs/quantization.md) |
| `docs/serving.md` | 上游服务接口与使用说明 | [服务说明](https://github.com/zyhector/token-rush/blob/b592bc6d91f0ea925b6a0179cbe65d1362b96381/docs/serving.md) |
| `docs/` 其余历史文件和图表 | 基线、旧环境、规划与性能图片 | [历史文档](https://github.com/zyhector/token-rush/tree/b592bc6d91f0ea925b6a0179cbe65d1362b96381/docs) |
| `bench/` 的历史评测入口 | 上游 decode、质量、接受率与长文本评测 | [原始评测工具](https://github.com/zyhector/token-rush/tree/b592bc6d91f0ea925b6a0179cbe65d1362b96381/bench) |
| `scripts/` 的跨框架和绘图脚本 | 原作者的基线、量化流程和图表生成 | [原始辅助脚本](https://github.com/zyhector/token-rush/tree/b592bc6d91f0ea925b6a0179cbe65d1362b96381/scripts) |

## 当前保留的辅助文件

| 文件 | 保留理由 |
|---|---|
| [model_card.md](model_card.md) | 模型格式与来源；其中的上游质量指标按原文保留 |
| [bench/quality_sources.py](../bench/quality_sources.py) | `tokenrush.convert` 仍导入其中的权重格式读取器 |
| [scripts/env_check/](../scripts/env_check/) | 现有环境说明引用的小规模 Triton、CUDA Graph、FLA 检查及设备诊断工具 |
| [data/quality/](../data/quality/) | 上游校准/评估 token 快照，供量化代码复查；不能在不同环境下保证重新生成逐字节相同的语料 |
| [tests/](../tests/) | 引擎数值与协议测试；本仓库实验工具的 CPU 测试另见 `experiments/tests/` |

这些保留项不代表 Swift Decode 已经重新做过上游的量化质量或服务实验。
本项目的结果以 [2026-10-03 实测报告](../reports/rtx5090-2026-10-03/) 为准，
讲解集中在 [项目详解与面试准备](SWIFT_DECODE_GUIDE_ZH.md)。
