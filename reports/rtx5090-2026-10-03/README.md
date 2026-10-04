# RTX 5090 单卡实验记录

当前主结果是 **gate-003 / bench-002**：在英文、代码、中文三条短输入、固定 128 token
的对照中，eager、CUDA Graph、MTP 输出逐 token 相同；MTP 相对 graph 的解码速度为
**2.00–2.26 倍**。完整配置、计时口径和波动见下表中的最新报告。

| 文件 | 用途 |
|---|---|
| [bench-002.md](bench-002.md) | 最新等输出性能报告，项目对外介绍引用这组结果 |
| [bench-002.summary.json](bench-002.summary.json) | 最新报告对应的云端原始汇总 |
| [bench-001.md](bench-001.md) | 首轮正常路径记录；MTP 输出有分叉，作为诊断历史保留 |
| [bench-001.summary.json](bench-001.summary.json) | 首轮原始汇总，不能与后续一致性结论拼接 |

本目录保存报告和汇总，不是完整 `runs/` 备份。逐请求 token、step 事件、环境和源码快照
以云端原始运行目录或其备份为准；本轮未执行独立 HF/BF16 精度验证。

操作步骤见 [一致性 benchmark](../../experiments/CONSISTENT_BENCH.md)，
问题定位见 [MTP 诊断记录](../../experiments/MTP_DIAGNOSTICS.md)，
学习与面试准备见 [项目详解](../../docs/SWIFT_DECODE_GUIDE_ZH.md)。
