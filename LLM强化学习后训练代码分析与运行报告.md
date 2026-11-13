# 大语言模型强化学习后训练的代码实现与机制分析

**基于 OpenRLHF 与 verl 的源码追踪及单卡验证**

本报告在后训练技术演进与方法比较的基础上，进一步开展强化学习后训练的源码分析与训练实践。研究选取 OpenRLHF 与 verl 两个开源项目，追踪 PPO、GRPO 的训练流程，并结合核心函数验证和小规模算术任务上的 SFT、GRPO 单卡实验，呈现分析路径、运行结果与实践收获。

## 摘要

报告围绕大语言模型强化学习后训练的实际程序展开分析，选取 OpenRLHF v0.8.0 和 verl v0.5.0 两个开源项目作为研究对象。前者重点考察带价值模型的 PPO 流程，后者重点考察不依赖独立 Critic 的 GRPO 流程。分析从启动脚本和配置项开始，沿着数据读取、回答生成、奖励计算、对数概率计算、优势估计、策略损失和参数更新逐层追踪，并将论文公式与源码中的张量运算对应起来。

两个项目的核心算法函数经过八组张量级测试，覆盖 PPO 的裁剪与回答掩码、终止奖励位置、GAE 回传、GRPO 分组归一化、损失归约和双重裁剪。此外，使用 Qwen2.5-0.5B-Instruct 与 LoRA 完成小规模算术任务训练，共调用优化器 48 次，测试集正确数从 3/32 增至 6/32。由于 48 个问题组中只有 5 组具有组内奖励差异，这一结果用于说明程序流程和算法机制，不用于比较算法性能。

针对初始实验中反馈稀疏的问题，随后进行了 SFT 热启动与 GRPO 分阶段实验。实验使用 2,048 道训练题、128 道验证题和 256 道新增测试题，并按验证正确率选择检查点。选中的模型在原测试集上答对 28/32，在新增测试集上答对 221/256（86.328125%）。继续 GRPO 后，新增测试集正确数为 220，验证成绩也未超过 SFT 检查点，因此最终保留了 SFT 检查点。此次提升来自监督热启动，后续强化学习更新未带来额外收益。

后训练程序的可靠性取决于损失函数、回答 token 边界、奖励位置、序列掩码、分组标识、损失归约方式及模型间的协同。PPO 通常需要 Critic、参考模型及多模型协同；GRPO 通过同一问题的多次采样构造相对优势。前者依赖价值估计质量，后者尤其依赖组内奖励差异，两者都要求可靠的奖励和评测。

**关键词：** 大语言模型；后训练；PPO；GRPO；强化学习；OpenRLHF；verl；源码分析

## 一、研究任务与分析范围

### 1.1 研究问题

研究结合源码追踪、张量级验证和单卡训练，考察以下四个问题：

1. 一条提示词从进入训练程序到引起模型参数变化，具体经过哪些模块？
2. 奖励如何从一个标量或一个 token 序列，变成可以反向传播的策略损失？
3. PPO 和 GRPO 在模型组件、优势估计、采样方式和资源消耗方面有什么差异？
4. 源码中哪些容易被忽略的实现细节，会改变算法的实际行为？

### 1.2 选择项目与方法

分析选取两个项目，分别追踪其训练调用链，并验证关键算法函数的行为。

| 项目 | 固定版本 | 重点方法 | 选择理由 |
|---|---|---|---|
| OpenRLHF | v0.8.0，提交 `a542dd98d29efacef89a4f694366e66c6e6d419b` | PPO、GAE、奖励模型和参考策略 | 具有较完整的 Ray 多角色训练流程，能够看到 Actor、Critic、Reference 和 Reward Model 的协作 [5, 8] |
| verl | v0.5.0，提交 `8fdc4d3f202f41461f4de9f42a637228e342668b` | GRPO、组内优势和策略更新 | 提供面向推理任务的分组采样与奖励处理，能够观察去除独立 Critic 后的实现方式 [6, 9] |

两个项目均保留了完整的源代码和提交版本。文件路径、函数名称和行号均以这两个固定提交为准。选取历史版本是为了固定可复查的实现范围，不代表项目的最新状态。OpenRLHF 的 PPO 与 verl 的 GRPO 分别代表两种优势估计思路：前者依靠价值模型估计优势，后者依靠同一问题的多条回答进行相对比较。基于人类反馈的语言模型训练提供了 PPO 后训练的典型背景 [3]，实际训练采用规则奖励。

verl 由字节跳动 Seed 团队发起并由开源社区维护，为分析国内开源项目中的分布式强化学习训练提供了具体实现 [9]。

### 1.3 研究边界

研究范围为语言模型后训练中的策略优化，不涉及预训练或奖励模型的完整训练。OpenRLHF 和 verl 的原生训练器涉及 Ray、分布式并行、vLLM 及多模型显存调度。实验采用核心函数级单卡程序，未运行面向 7B/8B 模型的原生多卡流程。源码验证考察算法行为，单卡训练验证前向、反向和参数更新。

## 二、环境、版本与材料

### 2.1 源码和模型

- OpenRLHF：<https://github.com/OpenRLHF/OpenRLHF>，v0.8.0，提交 `a542dd98d29efacef89a4f694366e66c6e6d419b`。
- verl：<https://github.com/verl-project/verl>，v0.5.0，提交 `8fdc4d3f202f41461f4de9f42a637228e342668b`。
- 基础模型：`Qwen/Qwen2.5-0.5B-Instruct`，下载清单记录的模型提交为 `7ae557604adf67be50417f59c2c2f167def9a775`。
- 初始实验适配器：LoRA，目标模块为 `q_proj` 和 `v_proj`，秩为 8，`lora_alpha=16`，对应的默认缩放比例为 `alpha/r=2`。
- 改进实验适配器：LoRA，覆盖注意力投影和前馈投影，秩为 16，`lora_alpha=32`；具体模块见第 8.5 节。

### 2.2 运行环境

| 项目 | 环境 |
|---|---|
| Python | 3.12.4 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.57.3 |
| PEFT | 0.17.1 |
| Accelerate | 1.10.1 |
| 随机种子 | 初始实验为 20261112，改进实验为 20261113 |

源码副本保存在 `advanced_artifacts/sources/OpenRLHF` 和 `advanced_artifacts/sources/verl`；源文件的 Git 提交和 SHA-256 清单保存在 `advanced_artifacts/results/core_verification.json`。实验配置、数据、日志、生成结果和 LoRA 适配器保存在 `advanced_artifacts/results/grpo_run`。

改进实验保存在 `advanced_artifacts/results/grpo_improved`，与初始实验分别存档；两次实验采用相同基础模型、任务定义和整数匹配评测规则。

## 三、源码分析路径

### 3.1 数据流梳理与函数追踪

分析先确定训练入口、数据流和模块边界，再阅读局部函数，核对输入输出张量及掩码。主要逻辑依赖如下，实际框架可以并行执行部分奖励和概率计算。

```text
启动脚本与配置
        ↓
训练入口和角色初始化
        ↓
问题数据、分词与批次构造
        ↓
当前策略生成回答
        ↓
奖励模型或规则函数打分
        ↓
当前策略、参考策略和 Critic 计算对数概率/价值
        ↓
奖励整形与优势估计
        ↓
策略损失、价值损失和 KL 项
        ↓
反向传播、优化器更新、权重同步和日志保存
```

记录各模块的入口文件、函数名称、输入输出张量及其对训练目标的影响，并使用小张量构造边界情况，核对源码结论与实际函数行为。

### 3.2 源码定位表

常用源码位置如下，行号以固定提交的源文件为准。

| 项目 | 文件 | 关键函数或类 | 作用 |
|---|---|---|---|
| OpenRLHF | `openrlhf/cli/train_ppo_ray.py:18` | `train` | 初始化 Ray、vLLM、Actor、Critic、Reference 和 Reward 角色 |
| OpenRLHF | `openrlhf/trainer/ppo_trainer.py:460` | `PPOTrainer.fit` | 生成样本、制作经验、写入经验池并触发 PPO 更新 |
| OpenRLHF | `openrlhf/trainer/ppo_utils/experience_maker.py:337` | `make_experience` | 汇总奖励、动作对数概率、参考对数概率和价值 |
| OpenRLHF | `openrlhf/trainer/ppo_utils/experience_maker.py:529` | `compute_advantages_and_returns` | 执行奖励整形、GAE 或 REINFORCE 类优势估计 |
| OpenRLHF | `openrlhf/trainer/ray/ppo_actor.py:195` | `training_step` | 计算策略损失、KL 和熵项并更新 Actor |
| OpenRLHF | `openrlhf/trainer/ray/ppo_critic.py:98` | `training_step` | 计算价值损失并更新 Critic |
| OpenRLHF | `openrlhf/models/loss.py:75` | `PolicyLoss` | PPO 概率比与裁剪目标 |
| OpenRLHF | `openrlhf/models/utils.py:43` | `compute_reward` | 将终止奖励放置到最后有效回答 token，并加入 KL 惩罚 |
| verl | `verl/trainer/main_ppo.py:44` | `run_ppo` | 初始化 Ray 并调度远程任务 |
| verl | `verl/trainer/main_ppo.py:91` | `TaskRunner.run` | 构建数据集、角色映射和 `RayPPOTrainer` |
| verl | `verl/trainer/ppo/ray_trainer.py:1084` | `fit` | 生成、打分、计算旧策略概率、优势并更新 Actor |
| verl | `verl/trainer/ppo/core_algos.py:246` | `compute_grpo_outcome_advantage` | 按问题 UID 分组并计算 GRPO 优势 |
| verl | `verl/trainer/ppo/core_algos.py:722` | `compute_policy_loss` | 概率比、裁剪和 dual-clip 策略损失 |
| verl | `verl/workers/actor/dp_actor.py:352` | `update_policy` | 微批次前向、损失归约、反向传播和优化器更新 |
| verl | `verl/workers/reward_manager/naive.py:44` | `NaiveRewardManager.__call__` | 解码回答并调用规则或奖励函数 |

### 3.3 公式与代码的对应方式

分析以论文公式为起点，追踪各变量在程序中的张量表示及其调用关系，重点核对以下三点：

1. 公式中的每一个量在代码中是否保留了相同的维度；
2. prompt token、padding 和回答 token 是否通过 mask 正确区分；
3. 代码是否加入了论文公式中没有直接写出的裁剪、数值稳定项或工程约束。

例如，PPO 的裁剪策略目标 [1] 可写为

$$
L_{\mathrm{policy}}=-\mathbb{E}\left[\min\left(r_t A_t,\;\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right],
\qquad r_t=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}}).
$$

在 OpenRLHF 的 `PolicyLoss.forward` 中，`ratio = (log_probs - old_log_probs).exp()`，随后分别计算 `surr1` 和 `surr2`，最后通过 `masked_mean` 汇总。verl 的 `compute_policy_loss` 先将对数概率差限制在 `[-20,20]`，再计算概率比，并额外实现了对负优势项的 dual-clip。仅看论文公式无法发现这两个实现差异。

### 3.4 关键张量及维度

设本轮输入问题数为 $B$，每题生成 $K$ 条回答，回答经 padding 后的宽度为 $T$，则送入优势和策略损失函数的样本数为 $N=BK$。以下维度针对普通的二维批次表示；序列打包和分布式切分会改变中间表示，但仍须恢复相同的样本与 token 对应关系。

| 张量 | 典型维度 | 含义及核对重点 |
|---|---|---|
| `responses` | `[N, T]` | 生成的回答 token，不含 prompt |
| `action_mask` / `response_mask` | `[N, T]` | 标记参与优化的回答位置，屏蔽后续 padding |
| 当前、旧策略和参考策略的 log probability | `[N, T]` | 三者须对应同一条回答的同一个 token；旧策略与固定参考策略作用不同 |
| `values`、GAE `advantages`、`returns` | `[N, T]` | PPO 的价值估计、优势和价值回归目标 |
| `token_level_rewards` | `[N, T]` | GRPO 输入的 token 奖励；结果奖励通常只在最后有效位置非零 |
| GRPO 的 `scores` | `[N]`，随后扩展为 `[N, T]` | 先沿 token 维求和，按 UID 中心化，再广播到有效回答位置 |
| `uid` / `index` | `[N]` | 同一问题的多条回答共享标识，重排时必须与其他张量同步 |

单卡实验中，$B=2$、$K=4$，每轮生成 8 条回答。若某条回答的有效长度为 3，奖励落点和损失分母仍应依据其 mask 计算，不能直接使用整批宽度 $T$。

## 四、OpenRLHF 的 PPO 实现分析

### 4.1 启动入口与模型角色

OpenRLHF v0.8.0 的入口是 `openrlhf/cli/train_ppo_ray.py:18` 的 `train`。该函数先建立 Ray 运行环境，再按参数创建 vLLM 生成引擎、Actor、Reference、Critic 和 Reward Model。`init_kl_coef` 大于零时才建立 Reference；使用 GAE 时需要 Critic，其他优势估计器则可以不建立 Critic。命令行参数在 `train_ppo_ray.py:441-450` 处进行约束：选择 `gae` 以外的分组或 REINFORCE 类估计器时，`critic_pretrain` 会被置空；`rloo`、`reinforce_baseline` 和 `group_norm` 还要求每个问题生成多于一个回答。

PPO 的一次训练循环涉及多个模型角色，其职责与更新方式如下：

| 角色 | 训练期间的职责 | 是否更新 |
|---|---|---|
| Actor | 生成回答并计算当前策略对回答 token 的 log probability | 是 |
| Reference | 提供基准策略的 log probability，用于 KL 约束 | 否 |
| Critic | 估计每个回答 token 的价值，供 GAE 使用 | 是 |
| Reward Model/规则奖励 | 为生成回答给出标量或序列奖励 | 取决于训练设置，PPO 主循环中通常只推理 |
| vLLM | 高吞吐生成 rollout | 不参与梯度更新 |

`PPOTrainer.fit` 位于 `openrlhf/trainer/ppo_trainer.py:460-518`。每一步依次取得 prompt，调用 `generate_samples` 生成回答，调用 `make_experience_list` 制作经验，将经验放入 Actor 和 Critic 的经验池，最后由 `ppo_train` 触发更新。Actor 更新完成后，`_broadcast_to_vllm` 将新权重同步给生成引擎。若持续遗漏权重同步，后续 rollout 会由陈旧策略生成，破坏当前更新与采样策略之间的对应关系。

### 4.2 动作概率与回答掩码

`openrlhf/models/actor.py:135-192` 的 `Actor.forward` 接收完整的 prompt+response 序列，但只返回回答部分的动作 log probability。代码先根据 `attention_mask` 计算 position id，使用滚动后的序列构造 next-token 标签，经过 `log_probs_from_logits` 后取最后 `action_mask.shape[1]` 个位置，并再次乘以 `action_mask`。

这一处理有两个直接后果：

- prompt token 不会作为 PPO 动作参与策略损失；
- 回答结束后的 padding 不会因为数值存在而影响损失和 KL。

若将整个拼接序列均视为 action，prompt 部分也会参与策略损失，改变预期的回答优化目标。因此，`action_mask` 直接界定参与优化的 token 范围。

### 4.3 经验制作与奖励整形

`make_experience` 在 `experience_maker.py:337-526` 中批量调用 Reward Model、Actor、Reference 和 Critic，最终为每个样本保存 `action_log_probs`、`base_action_log_probs`、`values`、`action_mask` 和 `reward`。当不使用独立 KL loss 时，代码在 `experience_maker.py:484-491` 计算当前策略和初始策略的近似 KL；当使用 KL loss 时，参考概率会保留到后续 Actor 更新。

`openrlhf/models/utils.py:43-71` 的 `compute_reward` 体现了序列奖励如何进入 token 级训练。若外部奖励为 $r_i$，代码先把它放在每个样本最后一个有效回答 token 的位置 $T_i$，再对有效回答 token 加入 KL 惩罚。这里的 $T_i$ 由 mask 决定，回答被长度限制截断时不一定包含真正的 EOS：

$$
R_{i,t}=\mathbf{1}[t=T_i]r_i-\beta\,KL_{i,t}.
$$

例如，一个长度为 2、外部奖励为 1、每个位置 KL 为 0.1、系数为 0.2 的回答，得到的奖励为 `[-0.02, 0.98]`。这与“把 1 平均分配给每个 token”完全不同。奖励位置会影响 GAE 的时间传播，也会影响长回答和短回答的梯度分布。

### 4.4 GAE、回报与 Critic 更新

在 `compute_advantages_and_returns` 中，`advantage_estimator == "gae"` 时调用 `get_advantages_and_returns`。该函数位于 `experience_maker.py:632-676`，按回答 token 从后向前计算：

$$
\delta_t=R_t+\gamma V_{t+1}-V_t,
\qquad
A_t=\delta_t+\gamma\lambda A_{t+1},
\qquad
G_t=A_t+V_t.
$$

其中的广义优势估计来自 GAE [2]。

实际代码先用 `action_mask` 将无效位置置零，再在倒序循环中构造 `delta`、`lastgaelam` 和 `returns`。随后，若不是 `group_norm` 或 `dr_grpo`，OpenRLHF 会在本轮汇总经验的有效 action token 上计算均值，默认再按标准差归一化；若开启 `no_advantage_std_norm`，则只减去均值。

Critic 的更新在 `openrlhf/trainer/ray/ppo_critic.py:98-141`。Critic 对相同的序列输出价值，`ValueLoss` 在 `models/loss.py:104-134` 中比较当前价值、旧价值和 returns；如果配置了 `value_clip`，先将当前价值限制在旧价值附近，再取未裁剪和裁剪平方误差的较大值。最后乘以 0.5，并使用回答 mask 汇总。这样做的目的，是防止价值网络在一次 PPO 更新中变化过大，破坏优势估计的稳定性。

### 4.5 Actor 更新和 KL

`openrlhf/trainer/ray/ppo_actor.py:195-259` 先用当前 Actor 重新计算回答 token 的 log probability，再调用 `PolicyLoss`，随后按配置叠加 KL 损失和熵项。

策略裁剪对应的原始代码如下，摘自固定版本 `openrlhf/models/loss.py:92-95` [8]：

```python
ratio = (log_probs - old_log_probs).exp()
surr1 = ratio * advantages
surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
loss = -torch.min(surr1, surr2)
```

这里的 `loss` 仍是回答 token 上的矩阵；函数随后才按 mask 归约为标量。`token_level_loss=True` 时所有有效 token 共同求平均，关闭时则先对每条序列求平均，再对序列求平均。因而 OpenRLHF 同样提供了不同的长度加权方式。忽略可选的辅助损失时，Actor 的总目标可写为：

$$
L=L_{\mathrm{PPO}}+\beta L_{\mathrm{KL}}-\alpha H(\pi_\theta).
$$

`compute_approx_kl` 支持 `k1`、`k2` 和 `k3` 近似。这里比较的是当前策略与固定参考策略：`k1` 使用两者的 log probability 之差，单个采样值可能为负；`k2` 是差值平方的一半；`k3` 使用指数形式构造非负估计。它们不应与 PPO 概率比中使用的旧策略混淆：旧策略对应本轮采样时的 Actor，固定参考策略用于限制整个后训练过程的偏移。KL 最终通过 `masked_mean` 在回答 token 上求平均。Actor 反向传播后调用优化器，更新完成后再把权重广播给 vLLM。

### 4.6 配置差异与资源调度

第一，OpenRLHF v0.8.0 已经把 PPO 的“优势估计”抽象成多个选项，`group_norm` 和 `dr_grpo` 可以在同一个工程中复用 Actor 更新逻辑。因而不能仅凭入口文件名把项目理解为只支持传统 PPO。

第二，`use_kl_loss` 会改变经验中参考概率的保留方式。代码在 `experience_maker.py:497-499` 中明确写出：不使用 KL loss 时，`base_action_log_probs` 会被删除。配置项不仅是开关，也会改变内存中的经验结构。

第三，原生脚本 `examples/scripts/train_ppo_llama_ray.sh` 面向 8B 级模型，配置了 Actor、Reference、Reward、Critic 和 vLLM 的多 GPU 资源，不能直接作为单卡实验命令。它适合用于理解工程资源布局。

## 五、verl 的 GRPO 实现分析

### 5.1 从 `main_ppo` 到训练循环

verl 的示例虽然使用 `main_ppo` 作为入口，但是否采用 Critic 由 `algorithm.adv_estimator` 决定。Hydra 入口 `main` 将配置传给 `verl/trainer/main_ppo.py:44-80` 的 `run_ppo`；后者初始化 Ray 并调度远程 `TaskRunner.run`。数据集、奖励函数和角色映射在 `TaskRunner.run` 中建立，随后在 `main_ppo.py:223-241` 初始化 `RayPPOTrainer`、调用 `init_workers`，最后进入 `fit`。

在 `verl/trainer/ppo/ray_trainer.py:372-384`，当估计器为 GAE 时设置 `self.use_critic = True`；当估计器为 GRPO、RLOO、REINFORCE++ 等时设置为 `False`。因此，GRPO 省略独立 Critic 并不是训练脚本的偶然写法，而是进入训练器后决定角色集合的显式逻辑。

### 5.2 rollout、奖励与 UID

`RayPPOTrainer.fit` 的一轮主要步骤位于 `ray_trainer.py:1156-1325`：

1. 将原始 batch 按 `rollout.n` 重复，并调用 rollout worker 生成回答；
2. 为原始问题生成 UID，再以相同的 interleave 顺序复制 UID，使同一问题的多条回答具有相同标识；
3. 构造 `response_mask`，计算奖励模型或规则函数的 token-level score；
4. 计算旧策略 log probability，必要时计算参考策略 log probability；
5. 根据配置决定是否把 KL 加入 reward；
6. 调用 `compute_advantage`，随后更新 Actor。

这一顺序解释了 GRPO 的关键数据条件：同一问题的多条回答必须在奖励计算和优势计算时仍然能够被识别为一个组。如果 UID 在 repeat 或 batch balance 过程中错位，模型比较的就不是同一个问题的候选答案。

`verl/workers/reward_manager/naive.py:44-120` 展示了规则奖励的具体接口。函数从 `prompts`、`responses` 和 `attention_mask` 中截取有效 token，解码成文本，再调用 `compute_score`。对于标量评分，代码把分数放在 `reward_tensor[i, valid_response_length - 1]`，其余位置保持为零。这与 OpenRLHF 的终止奖励放置方式一致，但两者位于不同的模块中。

### 5.3 GRPO 的组内优势

verl 的 `compute_grpo_outcome_advantage` 位于 `verl/trainer/ppo/core_algos.py:246-308`。它先对每条回答的 token-level reward 求和得到标量分数，再用 `index` 将样本分组。对于含 $K$ 条回答的第 $g$ 组，采用标准差归一化时，优势为：

$$
\mu_g=\frac{1}{K}\sum_{j=1}^{K}r_{g,j},
\qquad
\sigma_g=\operatorname{std}(r_{g,1},\ldots,r_{g,K}).
$$

$$
A_{g,j}=\frac{r_{g,j}-\mu_g}{\sigma_g+\varepsilon}.
$$

代码把这个标量优势扩展到整段回答，再乘以 `response_mask`。DeepSeekMath 提出了 GRPO，并将组内相对比较用于数学推理训练 [7]。该版本使用 `torch.std` 的默认样本标准差，因而手算对照时应采用相同口径。当 `norm_adv_by_std_in_grpo=False` 时，只减去组均值；这对应 Dr.GRPO 所讨论的归一化修改之一 [11]。仅修改这个开关，并不等于复现 Dr.GRPO 的全部设置，还需要核对长度归一化等损失配置。对于多样本组内分数完全相同的情况，中心化后的优势为零，这组样本不会通过策略项提供方向性信号。

这一过程可直接对应到 `core_algos.py:301-306` 的原始代码 [9]：

```python
for i in range(bsz):
    if norm_adv_by_std_in_grpo:
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    else:
        scores[i] = scores[i] - id2mean[index[i]]
scores = scores.unsqueeze(-1) * response_mask
```

`index[i]` 决定使用哪个问题组的统计量，`unsqueeze(-1)` 将每条回答的标量优势扩展为可广播的列向量，最后的乘法同时完成广播和掩码。返回的两个张量均为 `scores`；这里的第二个返回值不能解释成由独立 Critic 计算的价值目标。

对于只有一条样本的组，代码将均值设为 0、标准差设为 1，以避免除以零。计算时仍会除以 `std + epsilon`，因此回退后的优势与原始奖励存在微小差异。分组大小、UID 是否重复以及标准差归一化开关，都会影响最终的学习信号。

### 5.4 策略损失、dual-clip 与损失归约

`verl/trainer/ppo/core_algos.py:722-794` 中的 `compute_policy_loss` 与 OpenRLHF 的 `PolicyLoss` 都先计算概率比和 PPO 裁剪项，但 verl 还实现了 dual-clip：当优势为负时，用额外的 `clip_ratio_c` 对极端概率比进行限制。代码同时对 log probability 差值做 `[-20,20]` 截断，避免指数运算产生数值溢出。

verl 将损失矩阵交给 `agg_loss`（`core_algos.py:686-719`）。当前版本支持四种归约方式：

- `token-mean`：所有有效回答 token 共同求平均；
- `seq-mean-token-sum`：每个序列先求 token 总和，再对序列求平均；
- `seq-mean-token-mean`：每个序列先求 token 平均，再对序列求平均；
- `seq-mean-token-sum-norm`：该版本直接将有效损失总和除以 `loss_mask.shape[-1]`，并非再除以实际回答长度。源码注明，这个长度应尽量在训练中保持恒定；若随批次的 padding 宽度变化，损失尺度也会变化。

长度不同的回答在不同归约方式下会获得不同权重。因此，“使用同一个 GRPO 公式”并不意味着不同配置会产生相同更新。`verl/workers/actor/dp_actor.py:352-486` 还会在每个 PPO mini-batch 内拆分 micro-batch，累计梯度后再调用 `_optimizer_step`，这属于影响吞吐和显存的工程实现。

### 5.5 参考策略与 KL

verl 有两种常见的 KL 位置：`algorithm.use_kl_in_reward` 将 KL 加入 token-level reward，`actor.use_kl_loss` 则在策略更新时将 KL 作为附加损失。`ray_trainer.py:517-518` 会在两者同时开启时打印提示。Actor 更新中的 `kl_penalty` 位于 `core_algos.py:1034-1069`，支持 `k1`、`k2` 和 `low_var_kl/k3` 等形式。其中 `low_var_kl/k3` 不仅将参考与当前 log probability 的差限制在 `[-20,20]`，还将最终结果限制在 `[-10,10]`；因此极端输入下的行为也受数值裁剪影响。

实际使用时应明确只打开哪一种约束，或者说明为什么同时打开。否则，奖励端和损失端都会惩罚策略偏离参考模型，KL 强度会与只看配置文件时的直觉不同。

## 六、OpenRLHF PPO 与 verl GRPO 的比较

| 比较维度 | OpenRLHF PPO（GAE 配置） | verl GRPO |
|---|---|---|
| 主要反馈 | Reward Model 或规则奖励，加 KL 约束 | 规则/模型奖励，按同题回答分组比较 |
| 需要的模型 | Actor、Critic；启用 KL 时使用 Reference，奖励可来自 Reward Model 或规则函数 | Actor；启用 KL 时使用 Reference，GRPO 路径不启用独立 Critic |
| 优势来源 | 价值模型输出和 GAE | 同一 UID 组内奖励的均值、标准差 |
| 采样方式 | 可以每题一条或多条；多样本用于 RLOO、group norm 等变体 | 通常设置 `rollout.n>1`，依靠同题多条回答进行相对比较；单样本走回退分支 |
| 关键数据 | `values`、`old_action_log_probs`、`base_action_log_probs`、`action_mask` | `uid`、`token_level_rewards`、`old_log_probs`、`response_mask` |
| 策略限制 | PPO clip，可叠加 KL 和熵 | PPO clip、dual-clip，可叠加 KL loss 或 reward KL |
| 长度处理 | `action_mask` 和 `masked_mean` 控制有效回答 token | `response_mask` 加可选损失归约方式共同决定长度权重 |
| 主要计算代价 | Critic 前向、价值更新和多模型调度 | 同题多次生成带来的 rollout 成本 |
| 全对/全错组 | GAE 仍可根据价值和回报产生优势 | 组内奖励完全相同时，GRPO 策略优势为零 |
| 适用情形 | 奖励连续、价值估计有帮助的任务 | 结果可自动验收且同题候选之间容易产生差异的任务 |

比较应以具体的优势估计器与配置为单位。OpenRLHF 也支持 `group_norm` 和 `dr_grpo`，verl 也保留了 GAE、REINFORCE++ 等估计器。PPO 与 GRPO 的主要差异在于优势构造和模型角色配置，而两个框架均将多种估计器接入了可复用的 rollout 与 Actor 更新流程。

还需从资源消耗角度分析。GRPO 省略 Critic，减少了模型存储和价值前向，但为了获得组内相对奖励，需要对同一个问题生成多条回答。若回答很长、采样数很大，rollout 仍可能成为主要瓶颈。PPO 的 Critic 增加了训练组件和同步成本，但在奖励稀疏、连续或难以稳定形成组内差异的任务中，价值估计可能更有用。

## 七、核心函数的可重复验证

### 7.1 验证程序

验证脚本为 `advanced_artifacts/scripts/verify_core.py`。脚本通过 AST 仅加载固定提交中待分析的函数定义，不启动 Ray，也不修改上游源文件。执行前根据固定提交对应的文件清单核对源码 SHA-256；本地存在 Git 记录时，另行检查提交版本及已跟踪文件状态。提交附件仅保留源码快照和校验清单。命令为：

```powershell
python advanced_artifacts/scripts/verify_core.py
```

脚本运行结果写入 `advanced_artifacts/results/core_verification.json`，状态为 `all_passed`，共 8 组验证。实际函数输出及其对应的行为检查如下。

### 7.2 验证结果

| 验证项 | 设置与结果 | 说明 |
|---|---|---|
| PPO 裁剪和 mask | OpenRLHF 策略损失为 `-0.15`；最后一个被 mask 的位置梯度为 0 | 证明裁剪位置和回答掩码确实影响反向传播 |
| 终止奖励位置 | `r=[1,2]`、每 token KL 为 0.1、系数 0.2 时得到 `[-0.02,0.98]` 和 `[-0.02,-0.02,1.98]` | 外部奖励只放在最后有效回答 token |
| GAE | 单个末端奖励、`gamma=0.9`、`lambda=0.8` 得到 `[0.5184,0.72,1.0]` | 奖励从回答末端向前传播 |
| ValueLoss | 对设置的旧价值、当前价值和 returns 得到 `0.32` | 检查价值裁剪和 0.5 系数 |
| GRPO 分组与归一化 | 奖励 `[1,0,0,0]` 的组得到约 `[1.5,-0.5,-0.5,-0.5]`；关闭标准差后为 `[0.75,-0.25,-0.25,-0.25]`；打乱样本顺序后分组结果一致 | 同一组验证同时检查归一化开关和 UID 对齐 |
| GRPO 单样本回退 | 单条样本奖励为 7 时，优势约为 `6.999993` | 回退分支使用均值 0、标准差 1，且仍加入 epsilon |
| 损失归约 | token mean 为 `2.3333`，sequence mean 为 `2.0` | 有效 token 数不同会改变序列权重 |
| dual-clip | 同一负优势输入下 OpenRLHF 为 `4.0`，verl 为 `3.0` | 额外裁剪会改变极端概率比下的损失 |

对于单样本组，verl 的实现返回约 `6.999993` 而非严格的 7，这是因为代码采用 `std=1` 的回退值后仍除以 `std+1e-6`。这个差异虽小，边界分支仍需运行验证。

## 八、真实运行：初始 GRPO 验证与分阶段改进

### 8.1 实验目标和任务

实验采用一个可自动验收的算术任务：

> 计算 `a × b + c`，只输出整数结果。

每条回答的奖励为：整个去除首尾空白后的回答恰好是正确整数时取 1，否则取 0。训练集和测试集由同一随机种子生成，但使用不重叠的三元数组合。规则奖励不依赖人工评价。

数据构造范围为 `a∈[10,49]`、`b∈[2,9]`、`c∈[1,9]`，共 2,880 种组合。程序打乱全部组合后，取前 48 题作为训练集，再取 32 题作为测试集。训练集每题只进入一次 rollout，每次生成 4 条候选；测试集在更新前后均采用贪心解码，不参与奖励反馈和参数更新。

训练脚本为 `advanced_artifacts/scripts/run_grpo.py`。脚本加载 Qwen2.5-0.5B-Instruct [10]，加入 LoRA，只训练 `q_proj` 和 `v_proj`；rollout、奖励、GRPO 优势、verl 策略损失、KL 损失和 AdamW 更新均在单卡上执行。LoRA 通过低秩适配器减少参与更新的参数量 [4]。脚本调用固定版本 verl 中的 `compute_grpo_outcome_advantage`、`compute_policy_loss`、`agg_loss` 和 `kl_penalty`，保留这些函数的计算逻辑。实验完成了核心算法计算和真实参数更新，没有运行原生 Ray/vLLM 分布式训练流程。

### 8.2 实验配置

| 配置项 | 数值 |
|---|---:|
| 问题数/rollout step | 2 |
| 同题回答数 | 4 |
| rollout step 数 | 24 |
| 每个 rollout 的优化器调用次数 | 2 |
| 优化器调用总次数 | 48 |
| 最大回答 token 数 | 12 |
| 采样温度 / top-p / top-k | `1.0 / 1.0 / 0` |
| 学习率 | `1e-4` |
| 优化器 / weight decay / 梯度裁剪阈值 | `AdamW / 0 / 1.0` |
| PPO clip | `0.2` |
| dual-clip `clip_ratio_c` | `3.0` |
| KL 类型及系数 | `low_var_kl`，`0.01` |
| 损失归约 | `token-mean` |
| 可训练参数 | 540,672 |
| 基础模型与适配器的参数总数 | 494,573,440 |

基础模型使用 bfloat16 加载，LoRA dropout 为 0；参考概率由关闭适配器后的冻结基础模型计算，避免额外常驻一份完整模型。每轮先固定旧策略概率，再对同一批回答执行两次更新，期间不重新采样；KL 仅作为附加损失使用，不加入任务奖励。

回答 mask 包含首次出现的 EOS，屏蔽其后的 padding；若达到最大长度仍未生成 EOS，则保留全部已生成位置。因此，后文的平均回答 token 数包含实际生成的首次 EOS，不包含后续 padding。

命令为：

```powershell
python -u advanced_artifacts/scripts/run_grpo.py --steps 24 --output grpo_repeat
```

原始运行保存在 `results/grpo_run`；上面的复现命令使用新的输出目录，以保留原始记录。重复运行时应更换尚不存在的输出名称。

### 8.3 运行结果

| 指标 | 更新前 | 更新后 |
|---|---:|---:|
| 测试集规模 | 32 | 32 |
| 正确题数 | 3 | 6 |
| 正确率 | 9.375% | 18.750% |
| 格式有效率 | 100% | 100% |
| 平均回答 token 数 | 3.5625 | 3.59375 |

训练过程共生成 192 条回答，得到 24 个正确奖励，平均 rollout reward 为 0.125。48 个问题组中，39 组全部错误，4 组全部正确，5 组奖励有差异。只有这 5 组能通过 GRPO 的组内比较产生非零任务优势；其余全错组和全对组的组内优势为零。48 次优化器调用中有 34 次具有非零梯度，另有 14 次梯度为零。两种计数并不矛盾：任务优势为零时，独立的 KL 项仍可能产生梯度，AdamW 的历史动量也可能继续影响参数变化。因此，优化器步数和非零梯度次数都不能直接等同于有效任务反馈数量。

测试集中有 3 道题从错误变为正确，没有正确变错误的题目。LoRA 参数中有 540,671 个元素发生变化，参数变化的 L2 范数为 0.5679。训练耗时约 13.6 秒。这些指标仅反映固定种子、固定数据和短训练过程，不能据此宣称 GRPO 在一般数学任务上优于其他方法。

重新加载已保存的 LoRA 适配器后，32 条贪心输出均与更新后记录一致，正确数仍为 6。输出一致性记录在 `checkpoint_verification.json`。

### 8.4 失败样本和结果解释

更新后仍有回答错误，例如：

| 题目 | 正确答案 | 模型回答 |
|---|---:|---:|
| `27 * 8 + 5` | 221 | 235 |
| `32 * 9 + 4` | 292 | 280 |
| `19 * 3 + 5` | 62 | 58 |
| `26 * 2 + 6` | 58 | 62 |

这些错误表明，短程训练尚未使模型形成稳定的乘加运算能力。正确率的提高仅对应 3 道题由错误变为正确，现有结果不足以排除训练采样和有限测试集造成的波动。算法比较还需要增加数据、重复随机种子、设置独立基线，并在明确监督信息和计算成本的前提下比较 SFT、PPO 与 GRPO。当前运行用于验证代码流程，不据此判断算法优劣。

### 8.5 改进方案与数据划分

初始实验仅有 48 道训练题，且绝大多数回答组没有奖励差异。改进实验首先扩大训练数据并加入监督热启动，再在相同的正确/错误二值奖励下执行 GRPO。题目范围、输入模板、最大回答长度和贪心评测方式均沿用初始实验。全部正确答案由乘加运算直接计算，监督目标由确定性程序生成。

数据构造先排除原来的 32 道测试题，再使用固定种子打乱剩余组合。其中 256 道作为新增测试集，128 道作为验证集，2,048 道作为训练集，其余组合未使用。四个集合的三元数组合两两不重叠。训练开始前固定数据、训练轮数和模型选择规则：仅按验证集正确数选择检查点，分数相同时保留较早的模型。两套测试集均不参与反向传播、奖励反馈或检查点选择。原来的 32 题用于与初始实验对照，新增的 256 题用于扩大评测范围。新增测试集未排除初始实验的 48 道训练题，因此仅用于评价改进实验中从基础模型开始的各阶段，不用于评价初始 GRPO 适配器。

| 改进项 | 实际设置 |
|---|---|
| 训练/验证/原测试/新增测试题数 | 2,048 / 128 / 32 / 256 |
| LoRA 模块 | `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj` |
| LoRA 秩与 alpha | `r=16`，`alpha=32`，dropout 为 0 |
| 可训练参数 | 8,798,208 |
| SFT 训练预算 | 4 轮，每批 8 题，共 1,024 次优化器调用 |
| SFT 学习率 | 峰值 `5e-4`；前 5% 步数预热，随后余弦衰减至峰值的 10% |
| GRPO 训练预算 | 128 轮采样，每轮 2 题、每题 8 个候选，每轮更新 2 次 |
| GRPO 优化器调用/生成回答数 | 256 / 2,048 |
| GRPO 学习率与 KL | `2e-5`；`low_var_kl`，系数 `0.01` |
| 模型选择 | 每轮 SFT 后、每 32 轮 GRPO 后评估验证集 |

SFT 只对正确答案和终止符计算交叉熵，prompt 与 padding 不参与损失。GRPO 以验证集选中的 SFT 适配器为起点，并保留一份冻结的 SFT 参考适配器计算 KL；该阶段的参考策略不再是未经热启动的基础模型。两个适配器共享冻结的基础模型权重，以控制显存占用。GRPO 的优势、裁剪、损失归约和 KL 计算仍调用固定版本 verl 的原始函数。

复现命令如下，输出名称应尚不存在：

```powershell
python -u advanced_artifacts/scripts/optimize_grpo.py --output grpo_improved_repeat
```

### 8.6 分阶段结果与模型选择

| 阶段 | 原测试集（32 题） | 新增测试集（256 题） |
|---|---:|---:|
| 基础模型 | 3/32（9.375%） | 26/256（10.15625%） |
| 初始短程 GRPO | 6/32（18.750%） | 未评测 |
| SFT 热启动 | 28/32（87.500%） | 221/256（86.328125%） |
| SFT 后继续 128 轮 GRPO | 28/32（87.500%） | 220/256（85.9375%） |
| 按验证集选中的最终模型 | 28/32（87.500%） | 221/256（86.328125%） |

SFT 各轮的验证正确数依次为 61、80、106、115，验证集规模为 128。GRPO 在第 32、64、96、128 轮的验证正确数依次为 111、115、112、114。GRPO 未超过第 4 轮 SFT 的验证结果，按预设的同分保留较早检查点规则，最终选择 `sft_epoch_4`。最终适配器属于 SFT 热启动阶段，86.328125% 的正确率不能归因于 GRPO。

在新增测试集上，SFT 将 201 道错误题改为正确，同时使 6 道原本正确的题目变错，净增 195 道正确答案。继续 GRPO 修复 5 题、使 6 题由正确变错，正确题数净减 1。两阶段的总体正确率接近，但逐题输出并不相同；当前 GRPO 配置没有显示稳定的边际收益。改进实验各阶段在两套测试集上的格式有效率均为 100%。

改进实验总耗时约 446.9 秒，不含初始模型加载，计入了训练期间的验证和阶段评测。SFT 阶段约 210.1 秒，GRPO 阶段约 158.3 秒，两者均包含对应的验证过程。重新加载最终适配器后，原测试集 32 条和新增测试集 256 条输出均与保存前一致。

### 8.7 改进后的反馈分布与错误分析

改进实验的 256 个 GRPO 问题组中，210 组全部正确、6 组全部错误、40 组奖励有差异，有效差异组比例为 15.625%；初始实验为 5/48（约 10.42%）。改进阶段的平均采样奖励为 0.92285，256 次优化器调用均产生非零梯度，但验证正确数没有超过 SFT 检查点。大量全对组的奖励相同，仍无法提供组内区分信号。

| 题目 | 正确答案 | 初始 GRPO 回答 | 最终模型回答 |
|---|---:|---:|---:|
| `32 * 9 + 4` | 292 | 280 | 292 |
| `19 * 3 + 5` | 62 | 58 | 62 |
| `27 * 8 + 5` | 221 | 235 | 211 |
| `26 * 2 + 6` | 58 | 62 | 68 |

最终模型在原测试集仍有 4 道错误，新增测试集仍有 35 道错误。该任务只考察有限范围内的整数乘加，数据划分也仅按三元组组合区分；相同操作数或相近运算结构可能同时出现在不同集合中，因此结果属于同分布组合上的评测，不能推出对更大整数、未见算子或长推理任务的泛化能力。

改进实验同时增加了监督数据、训练预算和 LoRA 容量，无法据此区分各因素的独立作用。结果仅支持两点：完整改进方案优于初始短程运行；SFT 后继续 GRPO 未超过 SFT 检查点。各因素的贡献仍需在固定其余条件后逐项比较。

## 九、从源码得到的实现结论

### 9.1 奖励落点决定信用分配

OpenRLHF 的 `compute_reward` 与 verl 的规则奖励管理器都将标量结果放在最后一个有效回答 token。PPO 通过 GAE 的倒序递推将末端奖励传播到前面的有效 token；GRPO 则先获得序列标量优势，再直接广播到有效回答位置。KL 在每个有效位置计算，其最终权重仍取决于奖励处理和损失归约方式。训练时如果修改 EOS、padding 或 action mask，实际上是在修改信用分配，而不是单纯修改数据格式。

### 9.2 GRPO 的资源开销与有效采样

GRPO 通过同题多回答的相对奖励替代价值模型，减少了 Critic 的显存和前向开销。但组大小越大，生成成本越高；当大多数组全部正确或全部错误时，额外采样也未必提供更多学习信号。实际选择应同时考虑奖励差异率、回答长度、采样吞吐和 Critic 的价值估计质量。

### 9.3 损失归约会改变样本权重

`token-mean` 让长回答拥有更多有效 token，从而在总梯度中占有更大比例；`seq-mean-token-mean` 先对每条回答求平均，再让每个回答获得相近权重。对代码、数学推理或工具调用任务而言，回答长度通常与难度相关，因此损失归约不是无关紧要的实现选择，必须和评测指标一起记录。

### 9.4 KL 可以出现在两个不同位置

verl 明确区分 reward 中的 KL 与 policy loss 中的 KL；OpenRLHF 也会根据 `use_kl_loss` 改变经验制作方式。两者同时开启时，策略既在奖励阶段受惩罚，又在更新阶段受惩罚。分析训练日志时，应区分原始任务奖励、KL 惩罚、策略损失和总损失，不能只看一个汇总 reward。

### 9.5 分布式框架的正确性还包括调度正确性

OpenRLHF 需要在 Actor 更新后把新权重广播给 vLLM；verl 需要让 rollout 的 UID、旧策略 log probability、参考策略 log probability 和 response mask 在数据重排后仍然对应同一条回答。一个公式正确的单卡函数，如果在跨进程传输时丢失策略版本或样本索引，仍可能产生错误训练。后训练代码分析不能只停留在 loss 文件。

## 十、分析收获

源码追踪使奖励与参数更新之间的关系更加具体。`compute_reward` 和 `response_mask` 的实现表明，奖励落点、逐 token 的 KL 项以及无效位置的屏蔽方式共同决定实际更新。

OpenRLHF 与 verl 的对照显示，PPO 和 GRPO 共享回答生成、旧策略概率、参考策略概率、策略裁剪、损失归约和优化器更新等基础设施，主要区别在于优势构造和角色配置。verl 通过 `use_critic` 在训练器中明确处理这一差异。比较算法时，需要区分可复用的训练机制与具体方法的改动。

GRPO 分组依赖 UID、repeat 顺序和 batch 重排的正确对应。打乱样本顺序后，分组结果仍保持一致，说明优势计算依赖 UID，而非 batch 中的相邻位置。

小规模运行暴露出理论介绍中容易忽略的训练信号问题。初始实验中有 39 个全错组、4 个全对组，只有 5 组存在组内奖励差异。训练循环正常运行且参数发生变化，并不表示奖励信号足以支持有效学习；数据构造和验证器设计与优化算法同样重要。

改进实验进一步呈现了这一问题的两种情形：初始模型能力较弱时，回答容易集中在全错组；监督热启动后，GRPO 阶段又出现 210 个全对组。后者继续更新仍未超过热启动检查点。模型选择应依据独立验证结果，并记录收益来源与退化样本。

源码分析也明确了判断训练是否完成的依据。成功加载模型、生成回答或使 loss 下降，都不能单独证明参数发生了有效更新。需要同时检查反向传播、非零梯度、优化器步数、参数差异和独立测试集。本文未覆盖原生分布式训练流程。

## 十一、局限与后续改进

研究存在以下限制：

1. 实验只使用 0.5B 模型与 LoRA。改进阶段虽新增 256 道独立测试题，仍仅覆盖有限范围内的乘加组合，不能代表大规模推理任务。
2. 没有在本机启动 OpenRLHF 的 Ray 多角色 PPO，也没有启动 verl 的原生 Ray/vLLM 分布式 GRPO；源码结论来自固定版本阅读，运行验证来自核心函数级单卡程序。
3. 初始实验与改进实验各使用一个固定种子，没有进行同配置下的多种子重复；改进同时改变了数据、训练预算和适配器容量，不能分离各因素的独立贡献。
4. 奖励是严格的整数匹配，不能覆盖数学推理过程正确但格式不同、代码部分正确或工具调用中间步骤正确等情况。

后续应根据模型当前正确率调整训练题分布。初始阶段全错组较多，需要先建立基础能力；热启动后全对组占多数，则应在训练集内筛选或生成具有适当难度的问题，增加候选之间的奖励差异，同时记录额外筛选成本。进一步比较 SFT、PPO 和 GRPO 时，应固定数据划分、模型容量及预算口径，分别控制候选数、KL 系数和学习率，并开展至少三个随机种子的重复实验。测试集应始终用于最终评估，避免据其分数反复选择训练方案。

## 十二、结论

研究完成了两个后训练代码项目的源码分析，并沿着统一的数据流追踪了从回答生成到参数更新的过程。OpenRLHF 的 PPO 通过 Critic 和 GAE 构造优势，依靠策略裁剪、价值裁剪和 KL 约束维持更新稳定；verl 的 GRPO 通过 UID 组织同题回答，以组内奖励差异构造优势，在 GRPO 路径中省略独立 Critic，但增加了多次 rollout 的采样需求。

源码验证显示，回答掩码、终止奖励位置、损失归约、UID 分组和 KL 放置位置都会改变实际训练行为。单卡实验完成了生成、奖励计算、反向传播和 LoRA 参数更新；奖励同质化则使大多数 GRPO 组缺少方向性信号。后训练程序的可靠性需要同时考察算法公式、张量边界、数据关系、分布式同步和独立评测。

分阶段改进将新增测试集正确率从基础模型的 10.15625% 提高至 86.328125%，收益主要来自监督热启动；后续 GRPO 未超过验证集选中的 SFT 检查点。结果支持先诊断数据与学习信号，再依据独立验证选择训练阶段；强化学习在已有监督模型上继续训练并不保证取得额外收益。

## 参考文献

1. Schulman, J., et al. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347, 2017. <https://arxiv.org/abs/1707.06347>
2. Schulman, J., et al. *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438, 2015. <https://arxiv.org/abs/1506.02438>
3. Ouyang, L., et al. *Training Language Models to Follow Instructions with Human Feedback*. arXiv:2203.02155, 2022. <https://arxiv.org/abs/2203.02155>
4. Hu, E. J., et al. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685, 2021. <https://arxiv.org/abs/2106.09685>
5. Hu, J., et al. *OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework*. arXiv:2405.11143, 2024. <https://arxiv.org/abs/2405.11143>
6. Sheng, G., et al. *HybridFlow: A Flexible and Efficient RLHF Framework*. arXiv:2409.19256, 2024. <https://arxiv.org/abs/2409.19256>
7. Shao, Z., et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300, 2024. <https://arxiv.org/abs/2402.03300>
8. OpenRLHF contributors. *OpenRLHF source repository, v0.8.0*. GitHub, commit `a542dd98d29efacef89a4f694366e66c6e6d419b`. <https://github.com/OpenRLHF/OpenRLHF/tree/v0.8.0>
9. verl contributors. *verl source repository, v0.5.0*. GitHub, commit `8fdc4d3f202f41461f4de9f42a637228e342668b`. <https://github.com/verl-project/verl/tree/v0.5.0>
10. Qwen Team. *Qwen2.5-0.5B-Instruct*. Hugging Face model repository, revision `7ae557604adf67be50417f59c2c2f167def9a775`. <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct>
11. Liu, Z., et al. *Understanding R1-Zero-Like Training: A Critical Perspective*. arXiv:2503.20783, 2025. <https://arxiv.org/abs/2503.20783>

## 附录：复现实验所需文件

`advanced_artifacts` 目录包括：

- `scripts/source_loader.py`：校验源码快照并加载固定版本核心函数；
- `scripts/verify_core.py`：运行八组 PPO/GRPO 张量级验证；
- `scripts/download_model.py`：下载固定提交的基础模型并记录文件哈希；
- `scripts/run_grpo.py`：执行单卡 LoRA 参数更新；
- `scripts/optimize_grpo.py`：执行监督热启动及后续 GRPO 改进实验；
- `scripts/analyze_optimized.py`：核对改进实验的数据划分、阶段结果和奖励分布；
- `scripts/evaluate_saved.py`：重新加载适配器并核对评测输出；
- `README.md`、`requirements.txt`：环境安装和复现说明；
- `results/core_verification.json`：源码验证结果和文件哈希；
- `results/source_index.json`：固定版本的关键符号与行号；
- `results/upstream_source_manifest.json`：固定提交对应的源码文件与 SHA-256；
- `results/grpo_run/config.json`：实验配置、环境和模型版本；
- `results/grpo_run/training.jsonl`：48 次优化器调用日志；
- `results/grpo_run/evaluation_before.json`、`evaluation_after.json`：独立测试集结果；
- `results/grpo_run/rollouts.jsonl`：训练期间生成的回答与奖励；
- `results/grpo_run/checkpoint_verification.json`：32 条保存后输出的一致性检查；
- `results/grpo_run/adapter`：训练后 LoRA 适配器。
- `results/grpo_improved`：改进实验的配置、数据、分阶段评测、训练日志和最终适配器。

这些文件用于复查报告中的源码定位和运行数字，报告的实验规模和结论边界以正文为准。
