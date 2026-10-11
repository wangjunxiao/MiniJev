# Jev 类决策模型 —— 开源数据集调研

**调研日期:** 2026-09-29
**目标:** 为使用 tinygrad 训练的 Jev / MiniJev 决策模型（confidence + route 双头）寻找可用的开源数据集
**结论摘要:** GitHub / HuggingFace 上**不存在**直接给出 `local` / `rag` / `cloud` / `review` 四类标签的数据集。存在的是四类可**派生**出该标签体系的开源数据。本文给出 12 个可用数据集的清单、标签映射方案、许可证与落地路线。

---

## 0. 先明确：为什么没有现成的

Jev 的四路标签不是通用 NLP 任务标签，而是**部署架构标签**：

| 标签 | 决策含义 | 对应通用 NLP 标签 |
|---|---|---|
| `local` | 端侧小模型/规则可处理 | 简单意图（simple intent） |
| `rag` | 需要检索外部文档 | 知识密集 / open-domain QA |
| `cloud` | 需要复杂推理 | 复杂推理 / 代码 / 规划 |
| `review` | 高风险或无法判断 | 安全 / 越狱 / 拒绝 / 弃权 |

现有开源数据集覆盖的是**右列**，左列必须由你自己的映射规则派生。这意味着：
**不存在 `pip install datasets` 一行搞定的方案，但存在"下载 + 写映射脚本 + 跑通"一两天的方案。**

同时要避免的错误路径是：把某个数据集换个名字当成 Jev 数据集，然后继续报告 98% —— 那只是换了种方式的模板泄漏。

---

## 1. 类别一：路由决策数据集（最贴近 Jev 语义）

这类数据集直接就是「给 query 选一个后端」的监督数据，与 Jev 的路由头**完全同构**。

### 1.1 RouteLLM（lm-sys）

- 仓库: https://github.com/lm-sys/RouteLLM （Apache-2.0，5.6k stars）
- 论文: *RouteLLM: Learning to Route LLMs with Preference Data* arXiv:2406.18665
- 数据: 托管在 HuggingFace `RouteLLM` 与 `lmsys` 组织下

**数据构成**

| 部分 | 内容 | 规模 |
|---|---|---|
| 训练数据 | Chatbot Arena 人类偏好对，用 GPT-4 as a judge 增强 | 偏好对（弱模型 vs 强模型） |
| 校准数据 | 公开 Chatbot Arena 数据集 | ~33k 对话（`lmsys/chatbot_arena_conversations`） |
| 评测基准 | MMLU / GSM8K / MT-Bench | 标准基准 |

**标签形态**: `(prompt, strong_model_win_rate)` —— 连续值，表示「该换强模型的程度」

**映射到 Jev**

```
win_rate < 0.35  → local
0.35 ~ 0.65      → rag（需增强，但不必上最强模型）
> 0.65           → cloud
（高风险语义查询单独覆盖为 review）
```

**优点**: 标签是**真实人类偏好**，不是合成规则，天然具有上游可信度。
**风险**: 只区分「弱/强」两档，映射到四档时需要自己定阈值，且阈值会显著影响类别分布。

---

### 1.2 RouterBench（withmartian）

- 仓库: https://github.com/withmartian/routerbench （**MIT**，183 stars）
- 论文: *ROUTERBENCH: A Benchmark for Multi-LLM Routing System* arXiv:2403.12031
- 数据: HuggingFace（README 内 DATASET 链接）

**数据构成**

| 维度 | 内容 |
|---|---|
| Prompt 数量 | ~405k（prompt-level 记录） |
| 组成数据集 | 8 个基准（MMLU、ARC、GSM8K、HellaSwag、Winogrande、NaturalQuestions、OpenBookQA、TruthfulQA 等） |
| 候选模型 | 11 个 LLM，每个 prompt × 每个模型都有完整输出与打分 |
| 字段 | `prompt`、`model`、`completion`、`eval_score`、`cost`（prompt/completion tokens 计价） |

**为什么它对 Jev 特别有价值**: 每行是 `(query, model, score, cost)`，你可以**自己定义路由策略**再生成标签，例如：

```
对某 query，若最便宜模型分数已达标   → local
若需检索型小模型 / 中等模型才达标     → rag
若只有最强模型达标                 → cloud
若所有模型都低分或输出不安全         → review
```

这是**最干净的监督信号**，且完全离线可算 —— 不需要调用任何 API。

---

### 1.3 RouterEval（MilkThink-Lab）

- 仓库: https://github.com/MilkThink-Lab/RouterEval （EMNLP 2025 Findings，arXiv:2503.10657）
- 数据下载: Baidu Drive / Google Drive / HuggingFace（`get_router_dataset.py` 一键拉取）

**数据构成（本清单中规模最大的一个）**

| 维度 | 数值 |
|---|---|
| LASER 记录数 | **> 200,000,000** 条性能记录 |
| 覆盖 LLM | **> 8,500** 个 |
| 评测领域 | **12** 个（知识问答、常识推理、语义理解、数学推理、指令遵循等） |
| 核心内容 | `(query, llm_name, score)` 稀疏性能矩阵 |

**对 Jev 的用法**

RouterEval 把「路由」形式化成了**标准分类任务**：给定 query，预测最优模型。而你的 Jev 只需要把「最优模型档位」再聚合一次：

```
最优模型参数量/成本 ≤ 本地档   → local
最优模型属于检索增强档           → rag
最优模型属于旗舰档               → cloud
```

论文还提出 **model-level scaling up** 现象（候选池越大，好路由器的收益越大），正好是 Jev 的商业论证材料。

**优点**: 数据量巨大、完全离线、单卡可跑。
**风险**: 12 个领域里有大量与 Jev 无关的题（如 AIME 竞赛数学），需要先做领域过滤。

---

### 1.4 RouterArena（Rice University）

- 论文: *RouterArena: An Open Platform for Comprehensive Comparison of LLM Routers* arXiv:2510.00202
- 数据: HuggingFace `RouteWorks/RouterArena`
- 规模: **8.4k 问题**，来自 9 个领域、44 个 Dewey Decimal 类目

**对 Jev 最有价值的一点**: 该数据集**用 Bloom's Taxonomy 做了难度分层**。

| Bloom 层级 | 对应 Jev 路由 |
|---|---|
| Remember / Understand | `local` |
| Apply / Analyze | `rag` |
| Evaluate / Create | `cloud` |

**这是本清单里唯一自带「难度标签」且难度标签是教学法理论定义的公开数据集**，正好补上你 README「Next construction steps」里第 5 条提到的 complexity head。

---

### 1.5 LLMRouterBench（UIUC / 多机构）

- 仓库: https://github.com/ynulihao/LLMRouterBench （Findings of ACL 2026，arXiv:2601.07206）
- 数据: `bench-release.tar.gz` 预打包下载（Baidu Drive / Google Drive / HuggingFace）

**数据构成**

| 维度 | 数值 |
|---|---|
| 实例数 | **400,000+** |
| 数据集 | **21+**（性能导向 15 个 / 性能-成本导向 10 个） |
| 模型 | **33** 个（20 个 ~7B 轻量 + 13 个旗舰；GPT-5、Gemini-2.5-Pro、Claude-4、DeepSeek-V3.1…） |
| Tokens | ~1.8B |
| 字段 | `origin_query`、`prompt`、`prediction`、`ground_truth`、`score`、`prompt_tokens`、`completion_tokens`、`cost` |

**双重价值**:

1. **训练数据**: `cost` 字段让「成本感知路由」成为可监督任务 —— 直接对应 Jev 的降本主张。
2. **评测基准**: 它是目前最新、最严格的统一评测框架，可以用来打脸式地验证你的路由器是否真有收益。

**注意**: 仓库**不提供**合成数据生成脚本，数据全部是真实模型推理采集结果（约 1000 GPU 小时 + 3000 美元 API 成本）。这是它不可自己复现、只能下载的原因。

---

### 1.6 FrugalGPT（Stanford FutureData）

- 仓库: https://github.com/stanford-futuredata/FrugalGPT （Apache-2.0）
- 论文: *FrugalGPT* (TMLR 2024)

**构成**: Google Colab 笔记本 + **12 个商业模型的预生成 API 输出** + 评估脚本。
**实验任务**: 新闻分类、法律推翻判决检测、阅读理解。

**对 Jev 的价值**: 这是**级联（cascade）范式**的开源实现 —— 与 Jev 的「先端侧判断、不确定再上云」在架构上同源。它的 response scoring system（训练一个 DistilBERT/BERT 二分类器判断「当前回答是否足够好」）**就是 Jev 的 confidence head 的工业版**。

**可直接复用**: 它的 response-quality-classifier 训练数据构造方式（用 ground truth 自动标注「够好/不够好」），可以套到你的 confidence 头上，替代现在合成出来的 1.0 / 0.67 标注。

---

## 2. 类别二：意图 / 行为分类数据集（映射 local / rag / cloud）

这类数据集提供**细粒度意图标签**，你可以按「意图 → 是否需要外部知识 → 是否需要复杂推理」映射成三路。

### 2.1 CLINC150

- 获取: `load_dataset("clinc_oos", "plus")`（HuggingFace）
- 规模: 150 个意图类 + OOS（out-of-scope）类，~23k 条
- **Jev 用法**: 关键在 **OOS 类** —— 它天然对应 `review`（意图不可识别 → 人工审核）

```
150 个 in-scope 意图中的简单类  → local
需要外部知识/账户信息的意图类    → rag
OOS（out-of-scope）           → review
```

### 2.2 Banking77

- 获取: `load_dataset("PolyAI/banking77")`
- 规模: 77 个非常细粒度的银行客服意图，13,083 条真实客户 query
- **Jev 用法**: query 全部来自真实用户（非合成），表述自然、含大量口语与错字，正好是 Jev Dataset Spec 第 8 节要求但当前数据集完全缺失的**压力测试集**素材。

### 2.3 MASSIVE（Amazon）

- 获取: `load_dataset("AmazonScience/massive")`，论文 arXiv:2204.08582
- 规模: **> 1,000,000 条话语**，52 种语言，60 个意图 + 55 个槽位类
- **Jev 用法**: 1) 提供**中文**（zh-CN）子集，是当前 MiniJev 最缺的中文真实语料；2) 多语言覆盖可用于验证 `route` 判断是否跨语言稳定。

### 2.4 SNIPS / HWU64

- SNIPS: 7 个意图、~14k 条（语音助手域）
- HWU64: 64 个意图、~25k 条（多域）
- **Jev 用法**: 作为 CLINC150 的补充域，用于测试跨域泛化。

### 2.5 CLUE 中文分类集（TNEWS / IFLYTEK）

- 仓库: https://github.com/CLUEbenchmark/CLUE
- TNEWS: 短文本新闻分类（15 类）；IFLYTEK: 长文本应用描述分类（119 类，1.7 万+ 条）
- **Jev 用法**: 唯一现成的**中文多类目文本分类**标注数据，可用于构造中文 `local`/`rag` 样本，并压测模型在中文长文本与短文本上的差异。

---

## 3. 类别三：安全 / 拒绝数据集（映射 review）

MiniJev 的 `review` 类现在完全来自 5 条硬编码模板（"帮我处理一下"、"这个怎么办"…）。下面是**真正有标注质量**的替代来源。

### 3.1 WildGuard（Allen AI）

- 规模: ~92,000 条
- 标签: 三维 —— **harmfulness / refusal / compliance**
- **Jev 用法**: `harmfulness=True` → `review`；这是三分类里唯一**同时给出模型是否拒绝**的数据集，可直接监督「该不该转人工」这个判断本身。

### 3.2 FalseReject

- 获取: `FalseReject/FalseReject`（CC-BY-NC-4.0）
- 规模: 28.8k 条，覆盖 **44 个安全类目**
- 内容: **对抗性生成但实际无害**的 prompt + 语境感知的正确回答
- **Jev 用法**: 这是 Jev 最需要的**反向训练信号** —— 防止模型把任何带敏感词的 query 都判成 `review`。你的规则基线（55%）大概率就是被关键词误导，这份数据能直接量化并修正该偏差。

### 3.3 Aegis / ToxicChat（补充）

- Aegis: NVIDIA 安全数据集，MLCommons 维护
- ToxicChat: 真实用户与聊天机器人的有害对话（~10k）
- **Jev 用法**: 提供真实分布下的风险样本，避免安全类目全部是合成模板。

---

## 4. 类别四：检索需求数据集（映射 rag / cloud）

用于判定「是否需要外部知识」以及「检索是否足够」。

| 数据集 | 规模 | 语言 | 对 Jev 的用途 |
|---|---|---|---|
| **MS MARCO** | 15 万+ 训练样本 | en | 检索型 query 的标准来源；query 自然（真实 Bing 搜索） |
| **Natural Questions** | 79k train / 8.7k dev | en | 事实型 query，天然应走 `rag` |
| **HotpotQA** | ~113k | en | **多跳推理** → 关键：这是"需要检索 + 需要推理"的边界样本，正好卡在 `rag` 与 `cloud` 之间，是训练这两个类区分的核心素材 |
| **FRAMES** | 824 条 | en | 高难度多跳，人工校验，适合当 `cloud` 的极难子集 |
| **MS MARCO zh / DuReader** | 中文子集 | zh | 中文检索型 query |

**映射规则**

```
单跳事实查询 + 有明确文档来源  → rag
多跳 / 需跨文档综合 / 需推理    → cloud
无需任何外部知识即可稳定回答    → local
```

---

## 5. 汇总表

| # | 数据集 | 来源 | 许可证 | 规模 | 直接映射到 | 中文 |
|---|---|---|---|---|---|---|
| 1 | RouteLLM | lm-sys | Apache-2.0 | Arena 偏好对 | local/rag/cloud | ✗ |
| 2 | RouterBench | withmartian | **MIT** | 405k prompt × 11 模型 | 四类均可派生 | ✗ |
| 3 | RouterEval | MilkThink-Lab | 见仓库 | 2 亿+ 记录 / 8.5k LLM | 四类均可派生 | ✗ |
| 4 | RouterArena | RouteWorks (HF) | 见卡片 | 8.4k，Bloom 分层 | local/rag/cloud | ✗ |
| 5 | LLMRouterBench | UIUC 等 | 见仓库 | 400k+ / 33 模型 | 四类 + 成本 | ✗ |
| 6 | FrugalGPT | Stanford | Apache-2.0 | 12 模型输出 | confidence 监督 | ✗ |
| 7 | CLINC150 | HF `clinc_oos` | CC-BY-3.0 | 23k / 150+OOS | local/rag/**review** | ✗ |
| 8 | Banking77 | HF `PolyAI/banking77` | CC-BY-4.0 | 13k / 77 意图 | local/rag | ✗ |
| 9 | MASSIVE | HF `AmazonScience/massive` | CC-BY-4.0 | 1M / 52 语言 | local/rag | **✓ (zh-CN)** |
| 10 | WildGuard | Allen AI | 见卡片 | 92k | **review** | ✗ |
| 11 | FalseReject | FalseReject | CC-BY-**NC**-4.0 | 28.8k / 44 类目 | **review 纠偏** | ✗ |
| 12 | HotpotQA | HF | CC-BY-SA-4.0 | 113k | rag↔cloud 边界 | ✗ |
| 13 | CLUE (TNEWS/IFLYTEK) | CLUEbenchmark | 见仓库 | 1.7 万+ | 中文 local/rag | **✓** |

> 许可证提示: **FalseReject 与 APIGen-MT-5k 是 CC-BY-NC（禁止商用）**。Jev 产品愿景文档里写的是商业路线（开源 + 云服务），使用这类数据前必须确认合规，或只用于内部验证。

---

## 6. 推荐落地路线（一周）

### Day 1–2: 零成本全监督路由训练

1. 下载 **RouterBench**（MIT，最无风险）与 **RouterEval**。
2. 写一个转换脚本，对每条 query 计算「达标的最便宜模型档位」，生成 `route` 标签：

```
local  ← 最便宜模型即达标，且 query 长度短、无检索迹象
rag    ← 需中等模型或需检索才达标
cloud  ← 仅旗舰模型达标
review ← 全部模型低分 / 高风险类目命中
```

3. 用这个数据集跑一遍你现有的 `train_dataset.py`。**这是唯一能在真实数据上得到可信 Acc 的路径**，且不用改模型。

### Day 3–4: 补 review 与 confidence

4. 引入 **WildGuard** 生成 `review` 正样本，用 **FalseReject** 生成 `review` 负样本（对抗过度转人工）。
5. 用 **FrugalGPT** 的 response-quality 构造法，把 confidence 从「合成 1.0/0.67」换成「有 ground truth 的真实统计一致性」。

### Day 5: 中文与压力测试

6. 取 **MASSIVE zh-CN** 与 **CLUE/TNEWS** 构造中文子集。
7. 用 **Banking77**（真实用户 query）当压力测试集 —— 它的自然口语与错字正是当前合成数据完全缺失的。

### Day 6–7: 修复泄漏 + 重测

8. 在 `data/split.py` 中实现 `semantic_group_id` 分组划分（当前是按 route 分层随机切，导致 129 个测试 query 中 117 个与训练集重复）。
9. 用 **RouterArena** 的 Bloom 分层做难度切片评估，输出「简单/中等/困难」三档的 Acc，而不是一个整体数字。

---

## 7. 需要警惕的三件事

**1. 不要保留模板泄漏的划分。**
当前 `data/generate_dataset.py` 的模板填充 + `split.py` 的随机切分，会让同一 query 字符串同时落在训练与测试集（实测 129 个测试唯一 query 中 117 个重复）。建议先把这一步修掉，否则任何新数据集的引入都会被这个 bug 掩盖。

**2. 不要用「准确率」作为主指标。**
在四类均衡且含大量模糊样本时，Acc 会掩盖路由错误方向 —— 把 `cloud` 误判为 `local` 的代价远高于把 `local` 误判为 `rag`。建议改用**成本加权混淆矩阵**（misroute cost matrix），这才能对上 Jev 的产品论证。

**3. 不要跳过 licensing 检查。**
上表已标出所有已知许可证。商用前请逐个复核 HuggingFace 数据集卡片上的最新条款。

---

## 8. 参考链接

- RouteLLM: https://github.com/lm-sys/RouteLLM
- RouterBench: https://github.com/withmartian/routerbench
- RouterEval: https://github.com/MilkThink-Lab/RouterEval
- RouterArena: https://huggingface.co/datasets/RouteWorks/RouterArena
- LLMRouterBench: https://github.com/ynulihao/LLMRouterBench
- FrugalGPT: https://github.com/stanford-futuredata/FrugalGPT
- LLM 后训练数据集精选列表: https://github.com/mlabonne/llm-datasets
- CLUE 中文基准: https://github.com/CLUEbenchmark/CLUE
