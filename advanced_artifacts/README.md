# 源码分析与单卡运行附件

本目录配合正文《大语言模型强化学习后训练的代码实现与机制分析》使用，包含两个固定版本的源码、核心函数验证程序、初始与改进单卡训练脚本、日志和最终适配器。

## 1. 目录与运行范围

```text
advanced_artifacts/
├── scripts/                 验证、下载、训练、评测与材料核对程序
├── sources/OpenRLHF/        v0.8.0 源码快照与原项目许可证
├── sources/verl/            v0.5.0 源码快照与原项目许可证
├── results/
│   ├── core_verification.json
│   ├── source_index.json
│   ├── upstream_source_manifest.json
│   ├── base_model_manifest.json
│   ├── submission_audit.json
│   ├── grpo_run/            初始训练配置、数据、日志、评测和 LoRA 适配器
│   └── grpo_improved/       SFT 热启动及后续 GRPO 的配置、日志与最终模型
└── requirements.txt
```

运行分为两个层次：`verify_core.py` 直接验证固定源码中的算法函数；`run_grpo.py` 使用这些函数完成单卡生成、奖励计算、反向传播和参数更新。后者没有调用 OpenRLHF 或 verl 的原生 Ray/vLLM 分布式训练器。源码通过 AST 加载，函数体保持原样，只移除定义上的装饰器并延迟解析类型注解，以避开不参与核心计算的分布式导入。

目录保留两个固定提交的源码快照及已跟踪文件的 SHA-256 清单，供脚本校验源码内容。基础模型权重和本地依赖副本不包含在附件中，基础模型可按固定提交重新下载。

## 2. 安装环境

复现训练需要支持 CUDA 与 bfloat16 的 NVIDIA GPU；仅执行核心函数验证不需要 GPU。依赖版本见 `requirements.txt` 和下方安装命令。

以下命令在已激活的 Python 虚拟环境中、作业根目录执行。附件中的源码快照通过 SHA-256 校验，运行时不要求安装 Git；若使用自行克隆且保留 `.git` 的源码目录，则同时检查 Git 版本和文件状态。

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r advanced_artifacts/requirements.txt
```

复现实验所需的 PEFT 与 Accelerate 通过 `requirements.txt` 安装到虚拟环境；本地运行副本不包含在提交包中。

## 3. 验证核心函数

```powershell
python advanced_artifacts/scripts/verify_core.py
```

通过时，输出中应显示 `status: all_passed` 和 `probe_count: 8`。八组测试分别检查策略裁剪与掩码、终止奖励位置、GAE、价值损失、GRPO 分组与归一化、单样本回退、损失归约和 dual-clip。每次运行都会更新 `results/core_verification.json`。

固定版本对应的提交哈希如下：

- OpenRLHF：`a542dd98d29efacef89a4f694366e66c6e6d419b`。
- verl：`8fdc4d3f202f41461f4de9f42a637228e342668b`。

若源码目录缺失，可按上述提交从原始项目重新取得源码，并使用 `results/upstream_source_manifest.json` 检查文件内容；不要用最新分支替代固定版本。

## 4. 下载基础模型

```powershell
python -u advanced_artifacts/scripts/download_model.py
```

下载目标为 `Qwen/Qwen2.5-0.5B-Instruct`，固定提交为 `7ae557604adf67be50417f59c2c2f167def9a775`。脚本通过 `hf-mirror.com` 将所需文件下载至 `advanced_artifacts/models/Qwen2.5-0.5B-Instruct`，并生成 `download_manifest.json`。下载量约 1 GB；文件大小和 SHA-256 保存在 `results/base_model_manifest.json`，可用于校验。

## 5. 重新加载原始适配器

基础模型下载完成后，可先检查原始训练产物：

```powershell
python -u advanced_artifacts/scripts/evaluate_saved.py --run grpo_run
```

初始实验在 32 道测试题上的输出均与记录一致，答对 6 题，正确率为 0.1875，详情见 `grpo_run/checkpoint_verification.json`。不同计算设备、驱动或软件版本可能造成输出差异；脚本会逐题比较贪心生成结果，出现差异时报告错误。

## 6. 重新训练

```powershell
python -u advanced_artifacts/scripts/run_grpo.py --steps 24 --output grpo_repeat
python -u advanced_artifacts/scripts/evaluate_saved.py --run grpo_repeat
```

初始实验记录位于 `results/grpo_run`；重复实验可使用 `grpo_repeat` 或其他新名称。训练脚本不会覆盖已有的 `config.json`，每次运行需指定新目录。默认种子为 20261112；每轮使用 2 道题、每题生成 4 个回答，共进行 24 轮 rollout，每轮调用优化器 2 次。32 道测试题与 48 道训练题的三元数组合不重叠。

初始训练生成 192 条回答，执行 48 次优化器更新，其中 34 次梯度非零。测试集正确数由 3/32 增至 6/32；48 个问题组中只有 5 组的奖励存在差异。该实验用于检验代码流程，不足以比较算法性能。适配器可用于恢复最终模型并重新评测，但未保存逐步续训所需的优化器状态。

## 7. 复现改进实验

改进实验先对 2,048 道训练题执行 4 轮 SFT，再执行 128 轮 GRPO，分别对应 1,024 次和 256 次优化器调用。LoRA 秩为 16，覆盖注意力与前馈投影，训练参数量为 8,798,208。验证集 128 题、原测试集 32 题、新增测试集 256 题与训练集两两不重叠。固定验证规则为正确数最高者优先，同分保留较早检查点。

```powershell
python -u advanced_artifacts/scripts/optimize_grpo.py --output grpo_improved_repeat
```

默认种子为 20261113，完整记录位于 `results/grpo_improved`。训练程序不会覆盖已存在的输出目录。运行时从本地读取基础模型，不查询在线模型配置。

| 阶段 | 原 32 题测试集 | 新增 256 题测试集 |
|---|---:|---:|
| 基础模型 | 3/32 | 26/256 |
| SFT 热启动 | 28/32 | 221/256 |
| SFT 后最终 GRPO | 28/32 | 220/256 |
| 验证集选中的最终模型 | 28/32 | 221/256 |

最终选中第 4 轮 SFT 检查点。GRPO 阶段的验证正确数未超过该检查点，新增测试集正确数还由 221 降至 220，因此整体提升来自监督热启动。各阶段汇总见 `summary.json`，逐题输出见 `evaluation_*_test.json` 和 `evaluation_*_fresh_test.json`，检查点选择记录见 `validation_history.json`。

检查已保存的最终模型及改进实验记录：

```powershell
python -u advanced_artifacts/scripts/evaluate_saved.py --run grpo_improved
python advanced_artifacts/scripts/analyze_optimized.py
```

保存后，脚本会检查原测试集 32 道题和新增测试集 256 道题，288 条输出均与保存前一致。`analyze_optimized.py` 默认读取 `grpo_improved` 记录；分析重复实验时，可将脚本中的目录改为对应输出目录。初始实验与改进实验的配置不同，不能据此分离各项改动的独立作用。

## 8. 核对报告和附件

```powershell
python advanced_artifacts/scripts/audit_submission.py
```

核对程序检查引文编号、Markdown 结构、源码符号位置、文件哈希、数据划分、奖励、训练日志、评测数字和保存后输出记录，并将结果写入 `results/submission_audit.json`。

两个上游项目的源代码按各自许可证保留，许可证位于 `sources/OpenRLHF/LICENSE` 和 `sources/verl/LICENSE`。基础模型的使用条款见对应的公开模型仓库。
