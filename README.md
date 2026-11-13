# 大语言模型强化学习后训练代码分析

仓库包含 OpenRLHF 与 verl 的后训练代码分析、核心函数验证，以及小规模 SFT 和 GRPO 实验，重点讨论 PPO 与 GRPO 的数据流、优势估计、奖励处理、策略损失和参数更新。

## 实验结果

| 实验阶段 | 原测试集 | 新增测试集 |
|---|---:|---:|
| 基础模型 | 3/32 | 26/256 |
| 初始短程 GRPO | 6/32 | 未评测 |
| SFT 热启动 | 28/32 | 221/256 |
| SFT 后继续 GRPO | 28/32 | 220/256 |
| 验证集选中的模型 | 28/32 | 221/256 |

改进实验的主要收益来自 SFT 热启动。后续 GRPO 未超过验证集选中的 SFT 检查点，最终保留了该检查点。

## 文件结构

- `LLM强化学习后训练代码分析与运行报告.md`：源码分析、实验设计、结果和讨论。
- `advanced_artifacts/README.md`：环境安装、复现步骤和附件说明。
- `advanced_artifacts/scripts/`：源码验证、训练、评测和结果分析脚本。
- `advanced_artifacts/results/`：训练数据、配置、日志、评测记录及最终适配器。
- `advanced_artifacts/sources/`：固定版本的 OpenRLHF 与 verl 源码快照及其许可证。
- `advanced_artifacts/requirements.txt`：Python 依赖列表。

## 运行验证

在安装依赖后，于仓库根目录运行：

```bash
python advanced_artifacts/scripts/verify_core.py
python advanced_artifacts/scripts/audit_submission.py
```

`verify_core.py` 运行 PPO 与 GRPO 的核心函数测试；`audit_submission.py` 核对报告、源码位置、数据划分和实验记录。训练复现需要支持 CUDA 与 bfloat16 的计算设备。模型下载、训练和检查点评测步骤见 [`advanced_artifacts/README.md`](advanced_artifacts/README.md)。

基础模型可通过下载脚本按固定版本获取，Python 依赖见 `advanced_artifacts/requirements.txt`；源码快照保留上游许可证。
