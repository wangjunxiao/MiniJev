# MiniJev 代码地图与现状评估

> **⚠️ 历史文档（重构前快照）。** 本文记录的是 **2026-09-29 重构之前**的代码状态，用于留存问题证据与修复依据。
> §2「代码地图」、§3「运行链路」、§5 的 P1-4/P1-5/P2-6、以及 §6.2 的文件清单**都已过时**：项目已重构为
> `minijev/` 下 15 个模块 + `tests/`，旧的根脚本（`train_dataset.py`、`evaluate_baseline.py`、
> `evaluate_massive.py`、`examples/infer.py`、`train_toy.py`）与空的 `training/`/`inference/`/`utils/`
> 目录均已删除。**当前结构看 `README.md`；当前数字看 `PLAN.md` 的「状态」一节。**
>
> 本文仍然有效的部分：§1 项目定位、§5 的 P0-1 / P0-2 / P1-3（问题诊断与证据）、§6 的文档矛盾分析。

**生成日期:** 2026-09-29
**评估方式:** 全量源码通读 + 运行验证（lint / 训练 / 推理 / 基线 / 数据审计）
**结论摘要:** 代码可运行、风格干净、模型能收敛；真正的风险不在代码，而在**数据划分泄漏**、**confidence 头学不到东西**、**推理链路没接上权重**，以及**根目录规划文档与代码描述的是两个不同的系统**。

---

## 1. 项目定位

`MiniJev/README.md:3-14` 的定义是唯一与代码一致的定位：

> 一个 tinygrad 实现的极小决策模型：输入一条查询，输出 `confidence ∈ [0,1]` + 四路 `route`（local / rag / cloud / review）。它是**有监督的决策模型，不是文本生成器**。

### 与两篇源论文的关系（关键）

`MiniJev_Implementation_Plan.md:5` 声明 MiniJev 是两篇论文的最小实现：

| 源论文 | 核心方法 | 在当前代码中的落地程度 |
|---|---|---|
| **A**：arXiv:2503.23303《SalesRLAgent：实时**销售转化预测**的强化学习》 | GPT-4O 合成 120 万条 / 15 行业 / 3–27 轮对话；Azure 3072 维嵌入 + 参与度/SPIN/异议特征；状态编码器 + 策略网 + 价值网 + 元学习；CPU 6h 训练；96.7% / AUC 0.98；85ms、12 req/s | **完全没有落地**。代码中不存在 RL、不存在转化预测器、不存在 policy/value 网络 |
| **B**：arXiv:2510.01237《面向 LLM 可靠性增强的**置信度感知路由**》 | 生成前估 `P(幻觉\|Q,M)`；三信号 `C_sem=cos(P(h_final),e_ref)`、`C_conv=Var(前半)/Var(后半)`、`C_learned=φ(h_final)`，φ 为 4 层 MLP，权重由验证集学得；阈值 θ = 0.75 / 0.55 / 0.35；检测 0.42→0.74、FPR 0.09、F1 0.61→0.82 | **约 1/3 落地**。路由头落地了；三信号退化成单个 `Linear(64→1)`；θ 只剩 0.35 一个 |

两篇论文**同一位作者**（Nandakishor M），但六份根目录文档中**从未提及**这一点。

**这是理解本仓库的第一件事：MiniJev 实现了两篇源论文中约"半篇"的内容，而且丢掉的是论文 B 最核心的贡献（多信号置信度）。** 根目录文档里所有"转化 / RL / 三信号"的措辞都指向未落地的部分，只有 `MiniJev/` 里的代码代表真实进度。

---

## 2. 代码地图

### 2.1 目录结构（含实际文件数）

```text
D:\jev\MiniJev\
├── minijev/                    # 核心库（7 个文件，160 行）
│   ├── __init__.py             #   8 行  公开 API 汇总导出
│   ├── config.py               #  13 行  JevConfig：全部形状常量
│   ├── tokenizer.py            #  33 行  ByteTokenizer：UTF-8 字节级、无词表训练
│   ├── model.py                #  40 行  TinyJev：Embedding + masked mean pool + MLP + 双头
│   ├── router.py               #  29 行  Route 枚举 + 两条路由规则
│   ├── baseline.py             #  23 行  RuleBaseline：关键词下限
│   └── checkpoint.py           #  14 行  safetensors 存取
│
├── data/                       # 数据层（21 个文件）
│   ├── __init__.py             #   6 行
│   ├── schema.py               #  52 行  JevSample 数据契约 + JSONL 读写   ← 唯一权威契约
│   ├── split.py                #  37 行  按 route 分层切分               ← 缺语义组划分
│   ├── split_jsonl.py          #  19 行  split.py 的 CLI
│   ├── generate.py             #  58 行  玩具数据生成器（4 模板 × 4 条）
│   ├── generate_dataset.py     # 384 行  正式数据生成器（68 模板 + 9 组同义改写 = 1,036 条）
│   ├── inspect_dataset.py      # 102 行  数据集统计（硬编码绝对路径）
│   ├── train.jsonl             # 725 行  正式训练集
│   ├── val.jsonl               # 155 行  正式验证集
│   ├── test.jsonl              # 156 行  正式测试集
│   ├── toy.jsonl               #  32 行  玩具数据
│   └── splits/{train,validation,test}.jsonl  # 20 / 4 / 8 行，玩具数据的切分
│
├── models/                     # 训练产物
│   ├── minijev_phase1.safetensors      # 84 KB，20,933 参数   ← 有效权重
│   ├── minijev_phase1_metrics.json
│   ├── smoke.safetensors               # 84 KB，1 epoch 的遗留产物（哈希与上面不同）
│   └── smoke_metrics.json
│
├── examples/infer.py           #  40 行  单条查询 CLI                  ← 不加载权重
├── train_dataset.py            # 192 行  正式训练循环（Adam + 早停 + Macro-F1 选点）
├── train_toy.py                #  60 行  6 条样本的最小训练演示
├── evaluate_baseline.py        #  50 行  规则基线的 Acc / Macro-F1 报告
├── README.md                   # 项目唯一与现实一致的说明
├── MiniJev_Dataset_Spec.md     # 数据规范（路由标签定义、划分规则、验收标准）
├── Open_Source_Datasets_Survey.md  # 开源数据集调研（13 个数据集的映射方案）
├── pyproject.toml              # Ruff 配置（仅 lint，无依赖声明）
│
├── training/                   # 空目录 ← 与 README 源码布局冲突
├── inference/                  # 空目录 ← 与 README 源码布局冲突
└── utils/                      # 空目录 ← 与 README 源码布局冲突
```

### 2.2 依赖方向（无环，干净）

```text
                     minijev/config.py
                            │
              ┌─────────────┼─────────────┐
              ↓             ↓             ↓
      minijev/tokenizer  minijev/model  minijev/router
              │             │             │
              └──────┬──────┘             │
                     │                    │
              ┌──────┴──────┐             │
              ↓             ↓             ↓
        train_dataset  examples/infer  data/schema
        train_toy                      minijev/baseline
                                       evaluate_baseline
                     data/generate ──→ data/schema ──→ data/split
```

**唯一的跨层反向依赖**：`data/schema.py:8` 从 `minijev.router` 导入 `Route`。数据层依赖模型层，意味着数据契约与模型路由枚举耦合——想换路由集合必须同时改数据层。这是有意为之（单一真源），但要记录在案。

### 2.3 核心抽象一览

| 抽象 | 位置 | 职责 | 边界 |
|---|---|---|---|
| `JevConfig` | `config.py:5` | 冻结 dataclass，全形状常量 | `vocab=257, seq=128, embed=64, hidden=64, routes=4` |
| `ByteTokenizer` | `tokenizer.py:16` | UTF-8 字节 → id（`byte+1`，0 为 padding） | 无词表、无 OOV、确定性 |
| `TinyJev` | `model.py:19` | Embedding → masked mean pool → Linear+ReLU → 双头 | **无位置编码、无注意力、无序列结构** |
| `JevOutputs` | `model.py:12` | `confidence`（sigmoid 后）+ `route_logits` | |
| `Route` | `router.py:6` | 四路枚举 | `local / rag / cloud / review` |
| `route_from_logits` | `router.py:13` | route 头 argmax，`confidence < 0.35` 覆盖为 review | 阈值写死 |
| `route_from_scores` | `router.py:21` | 基于 complexity 的兼容路径 | **死代码**——不存在 complexity 头 |
| `RuleBaseline` | `baseline.py:6` | 关键词优先级匹配 | 下限参照 |
| `JevSample` | `schema.py:12` | `query / confidence / route / context` | 只读这 4 个字段 |

### 2.4 模型规模实测

```
参数量        20,933
权重文件      84 KB（vision 文档承诺 8MB，实际小 100 倍）
embed_dim     64（规划文档写 384）
hidden_dim    64
```

---

## 3. 三条运行链路

### 3.1 训练链路（已验证可跑通）

```text
data/{train,val,test}.jsonl
   ↓ data.schema.load_jsonl          # 只取 query / confidence / route
   ↓ ByteTokenizer.batch             # [B,128] int32 + [B,128] float32 mask
   ↓ TinyJev(input_ids, mask)
   ├─ route_loss      = sparse_categorical_crossentropy(routes)     # train_dataset.py:115
   └─ confidence_loss = MSE(confidence, target)                     # train_dataset.py:116
   loss = route_loss + 1.0 × confidence_loss                        # :117
   ↓ Adam(lr=3e-3) + 早停(patience) + 按验证集 Macro-F1 选点          # :124-141
   ↓ models/minijev_phase1.safetensors + _metrics.json
```

### 3.2 推理链路（**断的**）

```text
examples/infer.py
   ↓ TinyJev(JevConfig())     # ← infer.py:21 直接随机初始化
   ↓ route_from_logits
   输出
```

`examples/infer.py` **从头到尾没有调用 `load_checkpoint`**。README 自己承认这一点（`README.md:41`：初始权重是随机的，这条命令只检查执行路径）。**当前仓库不存在任何能加载 `minijev_phase1.safetensors` 并推理的入口。**

### 3.3 评估链路

| 脚本 | 输入 | 输出 |
|---|---|---|
| `train_dataset.py` 内置 `evaluate()` | val + test | Acc / Macro-F1 / 每类 P·R·F1 / 混淆矩阵 / CE / confidence MSE·MAE |
| `evaluate_baseline.py` | 任意 JSONL | 规则基线的同款指标（无 confidence 指标） |

---

## 4. 运行验证结果（本次实测）

| 检查项 | 命令 | 结果 |
|---|---|---|
| 代码风格 | `ruff check .` | ✅ All checks passed |
| 格式 | `ruff format --check .` | ✅ 18 files already formatted |
| 训练可跑通 | `train_dataset.py --epochs 3` | ✅ exit 0，26.9s，test Acc 0.8654 / Macro-F1 0.8587 |
| 推理可跑通 | `examples/infer.py "…"` | ✅ 输出 JSON，但 confidence=0.510、route=review —— **随机权重** |
| 规则基线 | `evaluate_baseline.py data\test.jsonl` | ✅ Acc 0.5577 / Macro-F1 0.5725 |
| 测试 | 无 | ❌ 全项目 **0 个测试文件** |
| 版本控制 | — | ❌ `D:\jev` 与 `D:\jev\MiniJev` **都不是 git 仓库** |

### 已训练模型（`models/minijev_phase1_metrics.json`）

| 指标 | 验证集 (155) | 测试集 (156) |
|---|---|---|
| Accuracy | 0.9806 | 0.9808 |
| Macro-F1 | 0.9786 | 0.9799 |
| Route CE | 0.0470 | 0.0395 |
| Confidence MAE | 0.1190 | 0.1189 |

best_epoch = 20，100 epochs 请求，147 秒，CL 设备。

**注意：这组 98% 的数字不可用于对外陈述，理由见 §5.1。**

---

## 5. 已核实的问题清单

### 🔴 P0-1 训练/测试集模板泄漏（阻断级）

实测审计（本次重跑确认，与上一轮会话记录一致）：

| 事实 | 数值 |
|---|---|
| 总样本 | 1,036 行 |
| **唯一 query 字符串** | **315** （平均每句重复 3.3 次，最多重复 5 次） |
| train 唯一 query | 297 |
| test 唯一 query | 129，其中 **117 与 train 重叠** |
| test 行级泄漏 | **140 / 156 行 (89.7%)** 的 query 文本已在 train 中出现 |
| 真正未见的 test 行 | **16 行** |

`MiniJev_Dataset_Spec.md:280-289` 明文禁止这种划分：

> 不能简单地按 JSONL 行随机切分相似样本 …… 这样可以避免同一模板同时出现在训练集和测试集中，导致数据泄漏。

`MiniJev_Dataset_Spec.md:349` 还把"测试集没有训练集的重复语义组"列为进入正式训练的**验收条件**。

**成因**：`data/generate_dataset.py:299-309` 的 `split_dataset()` 是 shuffle 后的**连续切片**，既不分层也不按语义组；`data/split.py:9` 虽然做了按 route 分层，但仍是逐行随机切分，且只被玩具数据用到。

**后果**：98.08% 衡量的是模板记忆。据上一轮会话在同一测试集上的切片测量，那 16 条未见样本上 Acc 仅 81.25% / F1 59.85%。

### 🔴 P0-2 confidence 头学不到任何东西

`data/generate_dataset.py:191-217` 的 `simulate_annotation()` 只产出两个 confidence 值：

| confidence | 行数 |
|---|---|
| 1.00 | 793 |
| 0.67 | 243 |

推导出的三个结论：

1. **标签里不存在 < 0.67 的 confidence**，而 `router.py:15` 的 review 覆盖阈值是 `< 0.35` —— 该分支在本数据集上**永不触发**，是死代码。
2. 最优常数预测器的 MAE ≈ 0.124，实测 confidence MAE = 0.119 —— **confidence 头只是在拟合 base rate，不含任何 query 信息**。
3. `MiniJev_Dataset_Spec.md:116-120` 定义了 3/3→1.00、2/3→0.67、1/3→0.33 三档，但 `simulate_annotation` **从不生成 0.33**，也从不触发 `:136` 要求的"三人完全不一致进仲裁"。

同时 `confidence` 的语义被混淆了：规范说它是"路由判断有多可靠"（标注一致性），而 `review` 标签是按**主题风险**（医疗/法律/金融/安全）判定的。两个不同概念被压进同一个头。

### 🟠 P1-3 `domain` 与 `risk_level` 是随机噪声

`data/generate_dataset.py:242` 与 `:271`：

```python
domain, risk_level = random.choice(domains)   # 只看 route，完全不看 query
```

实测：含"计算"字样的 query 被分到 `translation`(10)、`tech_documentation`(4)、`realtime_info`(1) 等域。`domain` 实际只是 `route` 的一个随机细分子标签，**不携带 query 的任何信息**。

规范 `MiniJev_Dataset_Spec.md:262-276` 说这些字段"用于统计、过滤、审计和错误分析"——按当前生成方式，用它们做错误分析会得到假的结论。`requires_retrieval` 同样由 `route in ["rag","cloud"]` 机械推出（`:253`），所以"写一篇技术博客"也被标成需要检索。

### 🟠 P1-4 推理链路没有接上权重

见 §3.2。仓库里没有任何加载 `minijev_phase1.safetensors` 的推理入口，`examples/infer.py` 永远是随机权重。一个用户按 README 跑 `infer.py`，看到的 confidence/route 与训练结果无关。

### 🟠 P1-5 零测试、零版本控制

- 0 个测试文件。`pyproject.toml` 里没有测试依赖，也没有 CI 配置。
- `D:\jev` 不是 git 仓库。整个项目（含 1,036 行标注数据、84 KB 权重、500 行文档）**没有任何版本历史**，一次误操作即不可恢复。

### 🟡 P2-6 卫生问题（不阻断，但影响可维护性）

| 问题 | 位置 |
|---|---|
| 三个空目录 `training/` `inference/` `utils/`，与 README 的源码布局矛盾 | `MiniJev/` |
| 硬编码绝对路径 `D:/jev/MiniJev/data/...` | `data/inspect_dataset.py:87-89`，`data/generate_dataset.py:371-373` |
| 陈旧产物 `smoke.safetensors` + `smoke_metrics.json`（1 epoch，哈希与 phase1 不同） | `models/` |
| 死代码 `route_from_scores()`（依赖不存在的 complexity） | `router.py:21-28` |
| `data/` 下两套并存的生成器：`generate.py`（玩具，README 指定为正式）与 `generate_dataset.py`（正式数据实际来源，README 却称其为"older exploratory"） | `README.md:28-30` 与 `data/train.jsonl` 实际来源矛盾 |
| 无依赖声明：`pyproject.toml` 没有 `dependencies`，靠 `PYTHONPATH=D:\jev\tinygrad` 手工注入 | `pyproject.toml` |

---

## 6. 文档 vs 代码：两份互相矛盾的规划

根目录 5 份文档（`Jev_Product_Vision.md`、`MiniJev_Implementation_Plan.md`、`MiniJev_Tech_Comparison.md`、`MiniJev_Quick_Reference.md`、`MiniJev_Project_Status.md`）描述的技术方案与 `MiniJev/` 里的代码**系统性不一致**：

| 维度 | 根目录文档的承诺 | 代码的实际状态 |
|---|---|---|
| 训练范式 | 强化学习（REINFORCE / PPO） | 有监督交叉熵，**无 RL** |
| 编码器 | LSTM，384 维 | 字节 Embedding + masked mean pool，**64 维，无序列建模** |
| 序列信息 | LSTM 隐藏态 | **无位置编码、无注意力**，是词袋 |
| 置信度 | 三信号加权（语义对齐 0.4 + 内部收敛 0.3 + 学习预测 0.3） | 单个 `Linear(64→1)` + sigmoid |
| 复杂度头 | `jev.predict_complexity()` 参与路由 | **不存在**（`route_from_scores` 是死代码） |
| 路由决策 | `confidence` 与 `complexity` 双阈值组合，4 场景 | route 头 argmax + 单一 `confidence<0.35` 覆盖 |
| 模型大小 | 8MB（V）/ ~40MB（T、Q）——**两份文档自己就差了 5 倍** | **84 KB** |
| 置信度阈值 | θ = 0.75 / 0.55 / 0.35 三档，配合 complexity 分 4 场景 | 只剩 `confidence < 0.35` 一档 |
| 置信度公式 | `P:144` 写 `0.3·conv`；`P:341` 写 `0.3·conv/10` 且 clip(0,10)；`T:211` 写固定权重 0.4/0.3/0.3 | 无公式，单个 sigmoid 头 |
| 隐藏维度 | `P:159` 写 256；`P:113/216`、`T:147`、`Q:72` 写 384 | 64 |
| 分流比例 | 文档内四套互斥：80/15/4/1、70/20/8/2、80/20、60/25/15 | 由 route 头自行学出，无预设比例 |
| Phase 1 定义 | `Implementation_Plan`：数据生成起步，共 6 阶段、10–16 天 | `README.md:16`：最小可执行决策模型（完全不同的一套边界） |
| 依赖 | numpy、tqdm、matplotlib | 只用 tinygrad + 标准库，`pyproject.toml` 无依赖声明 |

### 6.1 文档内部的自相矛盾（无需对照代码即可发现）

| 类型 | 具体 |
|---|---|
| 路线图冲突 | V 的 Phase 1 = 8MB 模型 + 80% 准确率，Phase 2 = 跨平台 SDK + 量化到 2MB；P 的 Phase 1 只是数据生成，共 6 阶段。V 每阶段 2–3 周，P 全流程 10–16 天 |
| 体积冲突 | V：8MB → 2MB，目标 <10MB；T:444 / Q:103：**~40MB** |
| 出口数量 | V 画了 5 个出口（含本地缓存）却称"4 路"；P/Q/T 只按置信度分 4 路 |
| 成本算术 | `V:38-43`：$0.11 × 1000 = $110/天，但结论写 $66/天；`V:176-181` 的 200+240 与所列百分比、$440 总额都对不上 |
| 延迟冲突 | `V:253-254` 手机 CPU 150ms vs `V:267-269` iPhone 45ms；笔记本 80ms vs Windows 120ms。指标 <200ms 与 Phase 2 的 <100ms 也不一致 |
| 无出处数字 | `V:67-68` 称"幻觉率 42%→15%、可靠识别 62%→92%"。论文 B 只有检测 0.42→0.74、F1 0.61→0.82，**15% / 92% 在论文中不存在** |
| 无据推测 | `P:90` "原论文：复杂的 PPO/A3C"、`T:13` "可能 PPO"、`T:9` 称论文 B 用 PyTorch。论文 A 只说"专门化 RL"，论文 B 只提 HuggingFace——**PPO / A3C / PyTorch 在原文中均未出现** |
| 参数算术 | `Q:74-86` 分项加总实为 ~5.4M 却写 8M，且 "0.16M × 2 = ~2M" 算错 |

### 6.2 承诺但从未创建的文件

文档给出的目录蓝图（`P:41-64`、`Q:96-98`）与实际磁盘的差距：

| 文档承诺 | 实际 |
|---|---|
| `data/synthetic_conversations.py`、`data/embeddings.py` | 不存在（实际是 `generate_dataset.py` / `generate.py`） |
| `models/{encoder,confidence_estimator,conversion_predictor,router}.py` | 不存在 |
| `training/train_{encoder,rl,confidence}.py`、`inference/pipeline.py`、`utils/*` | **目录存在但完全为空** |
| `examples/demo_*`、`embedder.pkl` / `predictor.pkl` / `confidence.pkl` | 不存在 |
| `training/train_all.py`、`examples/demo_cost_saving.py`、`deploy/export_mobile.py` | 不存在，`deploy/` 也没有 |
| `Q:266-267` 的 `git clone` | **从未发生**——`MiniJev/` 不是 git 仓库 |
| `S:17` 称已创建 `test_basic.py` | 实际只有 `D:\jev\test_tinygrad.py` |

### 6.3 关于技术选型论证

需要澄清一点：`MiniJev_Tech_Comparison.md` 与 `MiniJev_Quick_Reference.md` **只列出了 tinygrad 的正面理由**（<10k 行、易读、编译器可见、便于后期硬件优化），**没有评估或否决过任何替代方案**。PyTorch 只作为对照出现，TFLite / LangChain / DistilBERT 都只是竞品对比。所以"为什么用 tinygrad 而不是 PyTorch"在文档中**没有真正的决策记录**。

### 6.4 结论

根目录 5 份文档描述的是**一个从未被构建的系统**：两篇论文的完整蓝图（RL 转化预测 + 三信号置信度路由）。代码在半途改成了一个极小的有监督路由分类器，并且**选择了最小可行的那条路**——这在工程上是合理的收敛（`README.md:25` 明写 "no … reinforcement learning, complexity head, LSTM, or Transformer"），但文档没有跟着更新。

**因此这 5 份文档不能当作路线图使用。** 唯一与现实同步的是 `MiniJev/README.md` 与 `MiniJev/MiniJev_Dataset_Spec.md`。

`Jev_Product_Vision.md` 的**产品论证**（成本节省、幻觉抑制、端侧隐私）仍然有效且自洽，可以保留作为产品文档；其中所有**技术实现描述**已全部过期，且部分成本数字存在算术错误（§6.1）。

---

## 7. 下一步规划（建议）

排序原则：**先修测量工具，再谈模型能力。** 当前任何模型改进都会被 P0-1 的泄漏掩盖成"98%"，无法判断真假。

### 阶段 0：建立可信的测量基线（1-2 天）— 最高优先级

| # | 任务 | 验收标准 |
|---|---|---|
| 0.1 | `git init` + 首次提交（含数据与权重，或用 LFS） | 有可回滚的历史 |
| 0.2 | 实现按语义组的划分：`generate_dataset.py` 给**全部**样本（不只 36 条同义改写）分配 `semantic_group_id`；`data/split.py` 按组划分而非按行 | train ∩ test 的语义组 = 0；断言写进代码 |
| 0.3 | 加泄漏守卫测试（`tests/test_split.py`）：构造含重复模板的数据集，断言划分后无组跨集 | 测试存在且能失败 |
| 0.4 | 补 `tests/`：schema 往返、tokenizer 边界（128 字节截断、空串、纯 ASCII）、model 形状与 mask 全零不除零、router 阈值 | `pytest` 全绿 |
| 0.5 | 报告拆分：整体 Acc 之外，单列"未见语义组"切片指标 | 报告出现两个数字 |

**阶段 0 完成后的预期**：真实 Acc 会从 98% 掉到 80% 附近。这是**好消息**——它意味着测量终于可信了。

### 阶段 1：修复 confidence 的语义（1-2 天）

| # | 任务 | 验收标准 |
|---|---|---|
| 1.1 | 决策：confidence 是"标注一致性"还是"路由可靠度"？二者不能混。建议保留规范原义（一致性） | 写进 README |
| 1.2 | 让 `simulate_annotation` 产出三档（含 0.33），并对三人分歧样本走仲裁路径 | confidence 分布出现 0.33 |
| 1.3 | 把 review 覆盖阈值从写死改为**从验证集学出来**（README `:117` 已列为下一步） | 阈值由数据决定，有校准曲线 |
| 1.4 | 加 confidence 校准指标（ECE / 可靠性图），替代裸 MAE | 报告含 ECE |

### 阶段 2：接上推理链路（1 天）

| # | 任务 | 验收标准 |
|---|---|---|
| 2.1 | `examples/infer.py` 加 `--checkpoint`，默认加载 `models/minijev_phase1.safetensors` | 同一 query 的输出与训练后评估一致 |
| 2.2 | 加批量推理入口 + 延迟测量（对标 vision 的 <200ms） | 有实测延迟数字 |
| 2.3 | 删除或修复 `route_from_scores` 死代码 | 无未使用导出 |

### 阶段 3：换掉合成数据（依赖外部数据集，3-7 天）

`Open_Source_Datasets_Survey.md` 已给出可执行路线，本文不重复。要点：

- 不存在现成的 local/rag/cloud/review 标签集，必须派生（`Survey:5-23`）。
- 最干净的全监督路径：**RouterBench（MIT）+ RouterEval** 的 prompt-level 记录，离线计算"达标的最便宜模型档位"生成标签（`Survey:289-301`）——**不需要改模型代码**。
- `review` 正负样本：WildGuard + FalseReject（注意 FalseReject 是 **CC-BY-NC，禁商用**，与 vision 的开源+云服务商业模式冲突，`Survey:283`）。
- 中文覆盖：MASSIVE zh-CN、CLUE/TNEWS。

**决策点（需要你拍板）**：MiniJev 的目标是 (a) 中文路由，(b) 英文路由，还是 (c) 中英双语？这决定了先做哪个数据集，也决定 byte-level tokenizer 是否够用。

### 阶段 4：能力与部署（远期，等数据可信后再评估）

- README `:113-120` 列出的方向：两层微型 Transformer（**仅当 pooled 模型在上下文上失败时**）、complexity 标签。建议保持这个"仅在实验需要时才加"的克制。
- vision 文档的 Phase 2（量化到 2MB、iOS/Android 导出、<100ms）在当前 84 KB / 无导出链路的状态下**还不能开始**。

### 需要你决定的三件事

1. **文档收敛**：根目录 5 份过期文档是归档到 `docs/archive/`、还是改写成与代码一致？目前它们会持续误导任何新参与者（包括未来的我）。
2. **数据方向**：走 `Survey` 的 RouterBench 路线（英文为主、可全离线、MIT），还是先扩中文合成数据？
3. **模型路线**：接受"词袋 + mean pool"作为 Phase 1 的正式边界（README 的立场），还是按 vision 补 LSTM/Transformer 与 RL？

---

## 附：一页速查

```powershell
# 环境（每一步都需要）
$env:PYTHONPATH = "D:\jev\tinygrad"
cd D:\jev\MiniJev

# 质量门（当前通过）
ruff check . ; ruff format --check .

# 训练（实测 ~147s：best epoch 20 + 早停；本次 3-epoch 冒烟测试 27s）
python train_dataset.py --epochs 100 --batch-size 32 --patience 15

# 规则基线（Acc 0.5577 / Macro-F1 0.5725）
python evaluate_baseline.py data\test.jsonl

# 推理（⚠ 随机权重，未接 checkpoint）
python examples\infer.py "请总结这份合同的风险"

# 玩具链路
python -m data.generate --output data\toy.jsonl --per-route 25 --seed 7
python -m data.split_jsonl data\toy.jsonl --output-dir data\splits --seed 7
```

| 关键常量 | 值 | 位置 |
|---|---|---|
| vocab_size | 257 | `config.py:8` |
| max_seq_len | 128 | `config.py:9` |
| embed_dim / hidden_dim | 64 / 64 | `config.py:10-11` |
| num_routes | 4 | `config.py:12` |
| review 覆盖阈值 | 0.35（写死） | `router.py:15` |
| 默认 lr / batch / patience | 3e-3 / 32 / 15 | `train_dataset.py:183-188` |
| 参数量 | 20,933 | 实测 |
