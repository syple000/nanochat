# nanochat

[English](README.md) | 简体中文

![nanochat 标志](dev/nanochat.png)
![缩放定律](dev/scaling_laws_jan26.png)

nanochat 是一个极简的大语言模型（LLM）训练实验平台。它面向单个 GPU 节点设计，代码精简、易于修改，涵盖了大语言模型的所有主要阶段，包括分词、预训练、微调、评估和推理。例如，你只需花费 48 美元（在配备 8 张 H100 的 GPU 节点上运行约 2 小时），就能训练出自己的、能力达到 GPT-2 水平的大语言模型，然后通过简单的命令行界面与它对话。相比之下，2019 年训练 GPT-2 的成本约为 43,000 美元。如果使用竞价实例，总成本可以降至约 15 美元。更一般地说，nanochat 开箱即用，只需设置一个控制模型复杂度的参数，就能训练出一整个小系列的计算最优模型：`--depth`，即 GPT Transformer 模型的层数（达到 GPT-2 水平所需的深度大约是 26 层）。其他所有超参数（Transformer 的宽度、注意力头数、学习率调整、训练时长、权重衰减等）都会自动按最优方式计算。

如果你对这个仓库有疑问，我推荐使用 Devin/Cognition 的 [DeepWiki](https://deepwiki.com/karpathy/nanochat) 提问，也可以使用 [Discussions 讨论区](https://github.com/karpathy/nanochat/discussions)，或到 Discord 的 [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) 频道交流。

## 达到 GPT-2 水平的用时排行榜

目前，开发工作的重点是优化计算开销最大的预训练阶段。受 modded-nanogpt 仓库启发，为了鼓励进步和社区协作，nanochat 维护了一个“GPT-2 速通”排行榜，记录将 nanochat 模型训练到 GPT-2 水平所需的实际耗时，能力水平以 DCLM CORE 分数衡量。[runs/speedrun.sh](runs/speedrun.sh) 脚本始终提供训练 GPT-2 水平模型并与之对话的参考流程。当前排行榜如下：

| # | 用时（小时） | val_bpb | CORE | 说明 | 日期 | 提交 | 贡献者 |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 | - | 0.2565 | OpenAI 原始 GPT-2 检查点 | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 基线，略微超出计算最优训练量 | 2026 年 1 月 29 日 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 略低于计算最优训练量，**启用 fp8** | 2026 年 2 月 2 日 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | 将总批次大小增加到 100 万个 token | 2026 年 2 月 5 日 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | 将数据集更换为 NVIDIA ClimbMix | 2026 年 3 月 4 日 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [第一轮](https://x.com/karpathy/status/2031135152349524125) | 2026 年 3 月 9 日 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch 第二轮 | 2026 年 3 月 14 日 | a825e63 | @karpathy |

我们主要关注的指标是“达到 GPT-2 水平的用时”：在配备 8 张 H100 的 GPU 节点上，训练出 CORE 分数超过 GPT-2（16 亿参数）的模型所需的实际耗时。GPT-2 的 CORE 分数为 0.256525。2019 年训练 GPT-2 的成本约为 43,000 美元；得益于过去 7 年间整个技术栈的诸多进步，如今我们能以快得多的速度、远低于 100 美元的成本做到这一点，实在令人惊叹。例如，按当前每张 GPU 每小时约 3 美元计算，8 张 H100 的节点每小时约 24 美元，运行 2 小时约需 48 美元。

关于如何解读排行榜以及参与贡献，请参阅 [dev/LEADERBOARD.md](dev/LEADERBOARD.md)。

## 快速开始

### 环境配置

nanochat 使用 [uv](https://docs.astral.sh/uv/) 管理依赖。安装方法：

```bash
uv sync --extra gpu    # 适用于 CUDA（A100/H100 等）
uv sync --extra cpu    # 或者：适用于纯 CPU / MPS
source .venv/bin/activate
```

如果需要开发环境（额外安装 pytest、matplotlib、ipykernel、transformers 等）：

```bash
uv sync --extra gpu --group dev
```

### 复现 GPT-2 并与之对话

最有趣的体验莫过于训练自己的 GPT-2，然后与它对话。完成这一流程所需的全部步骤都包含在 [runs/speedrun.sh](runs/speedrun.sh) 这一个文件中，该脚本面向配备 8 张 H100 的 GPU 节点设计。从你喜欢的服务商那里启动一台新的 8 卡 H100 GPU 服务器（例如，我使用并且很喜欢 [Lambda](https://lambda.ai/service/gpu-cloud)），然后启动训练脚本：

```bash
bash runs/speedrun.sh
```

运行大约需要 1.5 小时，因此你可能希望在 screen 会话中执行。完成后，就可以通过命令行界面与你的模型对话。再次确认本地的 uv 虚拟环境已激活（运行 `source .venv/bin/activate`），然后开始聊天：

```bash
python -m scripts.chat_cli
```

让它写故事或诗歌。问它你是谁，看看它会怎样产生幻觉。问它天空为什么是蓝色的，或者为什么是绿色的。速通模型是在 4e19 FLOPs 的训练计算量下得到的模型，所以和它聊天有点像与幼儿园小朋友聊天 :)。下面是一段与速通模型的示例对话：

> **用户：** 你好，最近怎么样？
>
> **助手：** 你好！今天有什么我可以帮你的吗？
>
> **用户：** 天空为什么是蓝色的？
>
> **助手：** 天空呈蓝色，是因为一种叫作光的瑞利散射的视错觉。这种现象发生在阳光穿过空气中的微小颗粒时。这些颗粒由微小的蓝色和紫色颗粒组成，使光线以特定方式弯曲。
>
> **用户：** 真有意思。写一首关于它的诗吧。
>
> **助手：** 暮光之中，天空蔚蓝，<br>
> 一幅闪耀着蓝光的画布，天幕在此舒展，<br>
> 一个蓝色的世界，蓝色与更多的蓝色，<br>
> 交织成一幅如此精美的织锦，<br>
> 每一种色调都仿佛在吟唱一个故事。<br>
> ……

还有几点说明：

- 代码也能在采用 Ampere 架构、配备 8 张 A100 的 GPU 节点上正常运行，只是速度稍慢。
- 省略 `torchrun` 后，所有代码也能在单张 GPU 上正常运行，并产生几乎相同的结果（代码会自动切换为梯度累积），但等待时间会变为约 8 倍。
- 如果 GPU 的单卡显存小于 80GB，就需要调整一些超参数，否则会遇到 OOM（显存不足）。在脚本中找到 `--device-batch-size`，不断减小它，直到显存足够。例如，可以从默认的 32 减小到 16、8、4、2，甚至 1。如果这样仍然不够，就需要更深入地了解代码，并采用其他办法。
- 大部分代码都是比较常规的 PyTorch，因此应该能在支持 PyTorch 的平台上运行，例如 xpu、mps 等。不过，我没有亲自测试过所有这些代码路径，所以可能还有一些问题。

## 研究

如果你是一名研究者，希望帮助改进 nanochat，可以关注两个脚本：[runs/scaling_laws.sh](runs/scaling_laws.sh) 和 [runs/miniseries.sh](runs/miniseries.sh)。相关说明见 [1 月 7 日：miniseries v1](https://github.com/karpathy/nanochat/discussions/420)。为了快速实验（每次预训练约 5 分钟），我最喜欢的规模是 12 层模型（GPT-1 的规模），例如：

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

这会使用 wandb（运行名称为“d12”），只在最后一步计算 CORE 指标，不进行采样，也不保存中间检查点。我喜欢修改一些代码，再运行一次 d12（或 d16 等），看看是否有所改善，如此反复迭代。为了判断一次实验是否带来了改进，我通常会查看以下 wandb 图表：

1. `val_bpb`（验证损失，以每字节比特数为单位，不受词表大小影响）随 `step`、`total_training_time` 和 `total_training_flops` 的变化。
2. `core_metric`（DCLM CORE 分数）。
3. 显存使用率、`train/mfu`（模型 FLOPS 利用率）和 `train/tok_per_sec`（训练吞吐量）。

示例见[这里](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044)。

需要注意的是，nanochat 的代码和配置围绕一个控制复杂度的参数展开：Transformer 的深度。这个整数会自动决定其他所有超参数（Transformer 的宽度、注意力头数、学习率调整、训练时长、权重衰减等），使训练出的模型达到计算最优。其设计理念是，用户不必考虑或设置这些参数，只需通过 `--depth` 指定更小或更大的模型，其他一切都能自动配合。遍历不同的深度，就可以得到 nanochat 的一组不同规模的计算最优模型，也就是 miniseries。按照当前代码，达到 GPT-2 水平的模型（目前最受关注的目标）大致位于 d24–d26 范围。不过，任何准备合入仓库的改动都必须有充分的原理依据，能够适用于所有深度设置。

## 在 CPU / MPS 上运行

[runs/runcpu.sh](runs/runcpu.sh) 脚本提供了一个在 CPU 或 Apple Silicon 上运行的简单示例。它大幅缩小了待训练的大语言模型，使训练能在几十分钟这一合理时间范围内完成。以这种方式训练，无法获得很强的模型效果。

## 精度 / 数据类型（dtype）

nanochat 不使用 `torch.amp.autocast`，而是通过一个全局的 `COMPUTE_DTYPE`（定义在 `nanochat/common.py` 中）显式管理精度。默认情况下，它会根据硬件自动检测并选择：

| 硬件 | 默认 dtype | 原因 |
|----------|--------------|-----|
| CUDA SM 80+（A100、H100 等） | `bfloat16` | Tensor Core 原生支持 bf16 |
| CUDA SM < 80（V100、T4 等） | `float32` | 不支持 bf16；可通过 `NANOCHAT_DTYPE=float16` 使用 fp16（配合 GradScaler） |
| CPU / MPS | `float32` | 稳妥的默认设置。在较新的 macOS 上，MPS 也能正常使用 `NANOCHAT_DTYPE=bfloat16`（内存占用约减少 25%，速度相近） |

你可以通过 `NANOCHAT_DTYPE` 环境变量覆盖默认设置：

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # 强制使用 fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # 强制使用 bf16
```

工作原理：模型权重以 fp32 存储，以保证优化器的计算精度；自定义的 `Linear` 层则在前向传播时将权重转换为 `COMPUTE_DTYPE`。嵌入参数直接以 `COMPUTE_DTYPE` 存储，以节省内存。这样既能获得与 autocast 相同的混合精度收益，又能完全显式地控制各部分使用的精度。

注意：使用 `float16` 训练时，`base_train.py` 会自动启用 `GradScaler`，防止梯度下溢。监督微调（SFT）也支持这一机制，但强化学习（RL）目前还不支持。各处的 fp16 推理都能正常工作。

## 指南

我发布过一些可能对你有帮助的指南，以下按时间从新到旧排列：

- [2026 年 2 月 1 日：用远低于 100 美元的成本超越 GPT-2：nanochat 之旅](https://github.com/karpathy/nanochat/discussions/481)
- [1 月 7 日：miniseries v1](https://github.com/karpathy/nanochat/discussions/420)，记录了 nanochat 的第一个模型小系列。
- 如果想为 nanochat 增加新能力，请参阅[指南：数一数 strawberry 中的 r（以及增加能力的一般方法）](https://github.com/karpathy/nanochat/discussions/164)。
- [2025 年 10 月 13 日：nanochat 最初的介绍文章](https://github.com/karpathy/nanochat/discussions/1)。这篇文章介绍了 nanochat，不过其中一些信息如今已经过时，文中模型也比当前 master 分支上的版本旧得多，效果更差。

## 文件结构

```
.
├── LICENSE
├── README.md
├── dev
│   ├── nanochat.png
│   └── repackage_data_reference.py # 生成预训练数据分片
├── nanochat
│   ├── __init__.py                 # 空文件
│   ├── checkpoint_manager.py       # 保存 / 加载模型检查点
│   ├── common.py                   # 各类实用小工具，提升使用便利性
│   ├── core_eval.py                # 评估基础模型的 CORE 分数（DCLM 论文）
│   ├── dataloader.py               # 支持分词的分布式数据加载器
│   ├── dataset.py                  # 预训练数据的下载 / 读取工具
│   ├── engine.py                   # 利用 KV 缓存进行高效模型推理
│   ├── execution.py                # 让大语言模型通过工具执行 Python 代码
│   ├── gpt.py                      # 基于 nn.Module 的 GPT Transformer 模型
│   ├── loss_eval.py                # 评估每字节比特数（而非损失值）
│   ├── optim.py                    # AdamW + Muon 优化器，支持单 GPU 和分布式
│   └── tokenizer.py                # GPT-4 风格的 BPE 分词器封装
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # 模型小系列训练脚本
│   ├── runcpu.sh                   # 在 CPU / MPS 上运行的简单示例
│   ├── scaling_laws.sh             # 缩放定律实验
│   └── speedrun.sh                 # 训练成本约 100 美元的 nanochat d20
├── scripts
│   ├── base_eval.py                # 基础模型：CORE 分数、每字节比特数、采样
│   ├── base_train.py               # 基础模型：训练
│   ├── chat_cli.py                 # 对话模型：通过命令行界面聊天
│   ├── chat_eval.py                # 对话模型：任务评估
│   ├── chat_rl.py                  # 对话模型：强化学习
│   ├── chat_sft.py                 # 对话模型：监督微调（SFT）
│   ├── infer_bench.py              # 推理：延迟 / 吞吐量 / 显存基准测试
│   ├── tok_eval.py                 # 分词器：评估压缩率
│   └── tok_train.py                # 分词器：训练
├── tasks
│   ├── arc.py                      # 科学类选择题
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── gsm8k.py                    # 8 千道小学数学题
│   ├── humaneval.py                # 名称容易引起误解；实际是简单的 Python 编程任务
│   ├── mmlu.py                     # 涵盖广泛主题的选择题
│   └── smoltalk.py                 # 来自 Hugging Face 的 SmolTalk 综合数据集
├── tests
│   ├── test_attention_fallback.py  # FA3 / SDPA 注意力回退机制
│   ├── test_engine.py              # 推理引擎、KV 缓存
│   ├── test_execution.py           # 沙箱代码执行
│   ├── test_optim.py               # MuonAdamW 优化器（需要 GPU）
│   ├── test_tasks.py               # 任务切片、任务混合、HubDataset
│   └── test_tokenizer.py           # BPE 编码 / 解码往返、对话渲染
└── uv.lock
```

## 参与贡献

nanochat 的目标是推动微型模型的能力上限，让人们能够在不到 1,000 美元的预算内完成从头到尾的整个流程。易用性既涉及总体成本，也涉及理解上的复杂程度。nanochat 并不是一个事无巨细都能配置的大语言模型“框架”：代码库中没有庞大的配置对象、模型工厂，也没有复杂臃肿的 if-then-else 分支。它是一个统一、连贯、精简、易读、易改、极易 fork 的“强基线”代码库，旨在让你从头到尾运行整个流程，得到一个可以与之对话的 ChatGPT 模型。目前，我个人最感兴趣的是缩短达到 GPT-2 水平所需的时间，也就是让 CORE 分数超过 0.256525。目前约需 1.5 小时，已经从 3 小时降了下来；通过改进预训练阶段，我们还可以进一步缩短这一时间。

当前的 AI 使用政策：披露。提交 PR 时，如果某些部分有大语言模型的实质性参与，而且不是你亲自编写的，或你没有完全理解，请说明这些部分。

## 致谢

- nanochat 的名字源自我之前的项目 [nanoGPT](https://github.com/karpathy/nanoGPT)，后者仅涵盖预训练。
- nanochat 也受到 [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) 的启发。该项目通过明确的指标和排行榜，为 nanoGPT 仓库引入了竞赛机制。nanochat 在预训练方面借鉴了它的许多思路和部分实现。
- 感谢 [HuggingFace](https://huggingface.co/) 提供 fineweb 和 smoltalk。
- 感谢 [Lambda](https://lambda.ai/service/gpu-cloud) 提供本项目开发所需的算力。
- 感谢首席大语言模型魔法师 🧙‍♂️ Alec Radford 提供建议和指导。
- 感谢仓库大管家 Sofie [@svlandeg](https://github.com/svlandeg) 帮忙管理 nanochat 的议题、拉取请求和讨论。

## 引用

如果 nanochat 对你的研究有帮助，可以使用以下格式引用：

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## 许可证

MIT
