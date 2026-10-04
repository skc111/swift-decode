# Swift Decode

单张 RTX 5090 上的 27B INT4 推理复现与性能对照，围绕 **eager → CUDA Graph → MTP**
理解单流生成的执行成本、状态管理与输出一致性。

已完成环境验证、完整生成 gate、MTP 分叉诊断、GDN 一致性模式修正和匹配配置的性能实验。
当前主结果来自 2026-10-03 的 `gate-003 / bench-002`。

## 实测结果

Triton、BF16 KV、MTP depth 3；英文、代码、中文三条短输入，固定输出 128 token，
每组测量 3 轮。以下为解码 token/s 中位数，三种模式输出逐 token 相同。

| 输入 | eager | CUDA Graph | MTP | MTP / graph |
|---|---:|---:|---:|---:|
| 英文 | 66.37 | 88.88 | 177.54 | 2.00× |
| 代码 | 59.72 | 88.88 | 200.56 | 2.26× |
| 中文 | 53.67 | 88.81 | 188.30 | 2.12× |

速度按首 token 之后的 127 个实际输出计算，不含加载、编译和图捕获。
结论限于这组单请求工作负载；尚未执行独立 HF/BF16 精度或多请求服务验证。
配置、显存、首 token 时间、波动和原始汇总见 [最新实测报告](reports/rtx5090-2026-10-03/bench-002.md)。

## 从这里开始

| 目的 | 入口 |
|---|---|
| 理解项目、准备面试 | [项目详解与 40 个面试问题](docs/SWIFT_DECODE_GUIDE_ZH.md) |
| 复现已通过的 gate 和 benchmark | [一致性实验步骤](experiments/CONSISTENT_BENCH.md) |
| 理解那次 MTP 输出分叉 | [诊断记录与修正依据](experiments/MTP_DIAGNOSTICS.md) |
| 查看当前与历史结果 | [实验报告索引](reports/rtx5090-2026-10-03/) |
| 查环境与下载笔记 | [初次环境准备说明](experiments/README.md)；其中早期状态以本页及最新报告为准 |
| 查上游设计与历史工具 | [上游来源索引](docs/UPSTREAM.md) |

现有云端环境与模型已经准备完成。继续实验时沿用本地 checkpoint 和独立虚拟环境，
使用新的结果目录，先 gate 再测匹配配置；无需因整理目录重新安装或下载。

## 目录

```text
swift-decode/
├── tokenrush/       推理引擎、算子、状态、MTP 与可选服务实现
├── experiments/     当前单卡实验入口、诊断工具、协议说明和 CPU 测试
├── reports/         Swift Decode 实测报告与原始汇总
├── docs/            项目详解、模型卡、上游来源索引
├── tests/           引擎与协议测试
├── scripts/env_check/  环境和小规模数值检查
├── bench/           权重转换所需的格式读取器
├── data/quality/    上游量化校准/评估 token 快照
├── pyproject.toml   项目依赖
└── uv.lock          已使用的依赖锁定版本
```

本地 `runs/`、模型、下载缓存和生成产物不提交。已移出的上游历史日志、跑分工具、图表
和旧开发说明仍可通过 [来源索引](docs/UPSTREAM.md) 查看固定版本。

## 实现来源

底层引擎、量化与投机解码实现来自 [Token Rush](https://github.com/zyhector/token-rush)，
起点为 `b592bc6`。Swift Decode 新增实验框架、环境与源码记录、分叉诊断工具、
GDN 一致性模式的局部修正，以及 gate/benchmark 的配对输出核验。
上表仅引用本仓库实测；上游性能和质量数据由来源索引单独提供。
