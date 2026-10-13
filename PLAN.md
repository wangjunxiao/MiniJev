# MiniJev 方向决策与规划 v2

**日期:** 2026-09-29
**依据:** MASSIVE 零样本评估 + 新做的微调对照实验 + 两个独立调研（开源路由实现的模块划分 / RL 做路由的既有做法）
**一句话:** 架构没问题，测量有问题，RL 的缺口不是算法而是 reward。下一步做「诚实的数字 + 拆模块」，不做 RL。

---

## 状态（2026-09-29 → 10-11 持续更新）

### 当前轮（10-11）：偏好条件化的诚实结论 + 参考 [jev.kuhung.me](https://jev.kuhung.me/)

#### 参考资料：Jev 的社区分析（[jev.kuhung.me](https://jev.kuhung.me/) / [GitHub](https://github.com/kuhung/understanding-jev)）

这篇笔记给出了**目标里那句话的准确名字**：

> **RLCD（Reinforcement Learning from Calibration Decisions）**——训练目标不是语言质量，而是
> **ECE（期望校准误差）**：「确保模型输出 0.8 的置信度时，在统计上真实对应约 80% 的准确率」。

**这就是「类似于强化学习但又不完全是强化学习」**：对着一个**决策/校准目标**优化，而不是对
冻死的 reward 做策略梯度。这比我这轮做的「偏好条件化」更贴近目标本意，应该排在前面。

其余可直接对上 MiniJev 的点：

| 笔记内容 | MiniJev 现状 |
|---|---|
| 三种输出原语 `choice` / `score` / `noul` | **只有 `choice`**（4 路由），缺 `score` 与 `noul` |
| 单步前向、0 生成步、末位 logits 投影到候选槽位 | 一致（单头分类，无自回归） |
| 失败模式：**未校准过度自信**（Qwen 7B 预测错时 softmax 仍 >0.90，与官方一致性仅 73.8%） | **MiniJev 独立复现了同一失败**：MASSIVE 上 spread 0.9966 而 acc 0.18 |
| 失败模式：**无关项干扰**（加入「天气」选项会拉低原有选项的 log-odds） | MiniJev 是**固定分类头**，没有候选 token 投影，故**天然不受选项顺序/候选集污染影响**——这是架构上的一处真实优势，值得写进对比 |
| 失败模式：**缺少思维链**，单步无法做多层逻辑 | 与 hard benchmark 上 0.46-0.48 的表现一致 |
| KV Cache 前缀共享，几十个问题近乎零边际延迟 | 未实现（MiniJev 无缓存层） |

#### 偏好条件化：**只在中段生效，极值端失败**（负面，但已定位）

目标：一套策略服务一族权衡点。实测（同一冻结 checkpoint，5 个操作点）：

| cost share | 路由构成 | 与最优的一致率 |
|---:|---|---:|
| 0.00 | local 75% / cloud 25% | **19.8%** |
| 0.25 | local 75% / cloud 25% | 99.0% |
| 0.50 | local 76% / cloud 24% | **100.0%** |
| 0.75 | local 76% / cloud 24% | 75.7% |
| 0.90 | local 77% / cloud 23% | 75.7% |

**路由构成几乎不动。** 定位过程排除了三个可能，每个都有独立证据：

1. **不是数据错。** `preference_targets` 在 share 0 上的目标分布是 local 20% / **review 80%**，
   与 oracle 一致。
2. **不是任务不可学。** 用**同样的 share-0 标签**训一个**普通模型（无偏好输入）**，
   **第 10 个 epoch 就到 val_acc 1.0000**。编码器和数据都没问题。
3. **不是「加法条件化」的锅**（虽然那确实是个真缺陷，已修，见下）。

**结论：是多任务干扰。** 一个共享编码器 + 一个头要同时服务 7 个操作点，中段样本密、极值端样本疏，
极值端输了。这不是能靠调阈值或加 epoch 解决的问题，需要**按操作点分配容量**，或者大幅加密极值端采样。

#### 过程中修掉的两个真设计缺陷

| 缺陷 | 症状 | 修复 |
|---|---|---|
| `difficulty_of` 用 **SHA-256 哈希**文本 | 难度与文本内容无关 → 最优路由**原理上不可预测**，模型只能学多数类（表现为 val_macro_f1 0.16 后衰减，**看起来像欠拟合**） | 改为**可观测**表面特征（长度 + 子句标记）的显式函数 |
| 偏好只用 `Linear([h; p])` 拼接 | `W_p·p` 是**每路由一个常数、与 query 无关**，只能平移全局 bias，**永远无法真正改道** | 加交互层：偏好与 query 在**非线性之前**相遇 |

#### 对我自己上一轮结论的更正

上一轮我写「条件化只花 8 个参数（<0.01%），不构成容量混淆」。**加了交互层后这句话是假的**：
代价是 `hidden × (hidden + preference_dim)` = **4224 参数，约 +20%**。测试已改为断言**真实代价**，
并明确写「这不是可忽略的量，条件化模型胜过无条件模型时不能全归因于偏好」。

#### 下一步（按目标本意排序）

1. **RLCD 式的校准训练**——这才是「像 RL 但不完全是 RL」的正解，且 MiniJev 的过度自信问题已复现。
2. 补 `score` / `noul` 两种原语（架构上与 Jev 对齐）。
3. 偏好条件化：按操作点分配容量，或加密极值端采样。
4. 基准侧：`implied` 68 组对比对，瓶颈仍是 `given_text` 对象数。


### 当前轮（10-11）：reward 层对齐文献 + 基准可解释化

按目标「reward 构建 + 多找参考资料」做了一轮。**参考资料确实改变了实现**，不是走过场。

#### 参考：BaRP（arXiv 2510.07429）

[Learning to Route LLMs from Bandit Feedback: One Policy, Many Trade-offs](https://ar5iv.labs.arxiv.org/html/2510.07429)
——把路由建模成**带偏好的上下文老虎机**，与「像 RL 但不完全是 RL」这个定位高度吻合。三条可搬的东西：

1. **reward 形式** `r = w_q·q − w_c·min(c/τ, 1)`。我的 `cost_scale` 已经等于 τ（归一化），
   但**缺了 cap**。没有 cap 时人类审核候选成本 2.0 对模型 0.1，成本项会独自决定比较。
2. **偏好向量在 1-simplex 上**，训练时均匀采样，推理时按需给定 → **一套策略服务多个权衡点，无需重训**。
   这是「多目标路由」的核心，我此前完全没有。
3. REINFORCE 用**熵正则 + 批均值 baseline**；我只实现了 baseline，没有熵项。

#### 落地

| 项 | 内容 |
|---|---|
| `RewardWeights.cost_cap` | 归一化成本封顶，**默认 None 保持 `utility-v2` 数字可复现** |
| `Preference` | 1-simplex 上的质量/成本权重，`at_cost_share()` / `normalized` / `cost_share` |
| `RewardWeights.with_preference()` | 只改两个权重，其余（scale/cap/error_costs）不动 |
| `PREFERENCE_SWEEP` | 0 / .25 / .5 / .75 / .9 五个权衡点 |
| `reward_baselines()` | oracle（每问最优）+ 四个 always-route + `per_query_best_route` |
| `regret()` | `headroom_captured` = 相对「最佳常数策略」到 oracle 之间捕获了多少 |

#### 我自己写的新代码里的一个真 bug（已修）

偏好扫描第一次跑出**负 headroom**——oracle 低于 floor，数学上不可能。调试发现：
常数策略**只在它服务得了的行上计分**，于是 `always_rag` 跳过了它无法服务的医疗行（该行 −0.600），
总分反而**超过** oracle。

**一个会挑行弃权的基线不是基线。** 修复：无法服务某行时按**终态 review 兜底**计分（部署系统本来
就会这么做，env 已经会合成 review），并在函数里**断言** `oracle ≥ best_constant`——违反它是 bug
不是发现。修完后 headroom 全为正且随成本权重单调递减（1.0000 → 0.7197 → 0.4645 → 0.2317 → 0.0860），
这才自洽。

这个 bug 值得记住：**它长得像一个结论**（「成本权重上升时 headroom 变负」完全可以是真发现），
是「oracle 不可能低于 floor」这条硬不变量把它揪出来的。

#### 实测（RouterBench fixture，4 行）

```
uncapped:  oracle 3.5950  floor 1.9045 (always_rag)
capped@1:  oracle 3.6450  floor 2.6545 (always_rag)
decisions changed by the cap: 0/4
```

**cap 在这 4 行上只动了算术，没动任何决策**——诚实地说，fixture 太小，无法证明 cap 的价值。
它的理由来自文献与量纲分析（2.0 vs 0.1），不是来自这份数据。

#### 偏好向量已成为模型输入（本轮完成）

`JevConfig.preference_dim`（默认 0，旧 checkpoint 全部照常加载）。> 0 时 route head 读
`hidden_dim + preference_dim`：

```
偏好差异 → logits 真的不同（不是装饰）
route_head.weight shape = (4, 66) = hidden 64 + preference 2
四种接线错误全部拒绝：普通模型喂偏好 / 偏好模型不给 / 宽度错 / 批量不匹配
```

**拒绝而不是静默兜底**是刻意的：静默替换成默认值会让一次**无条件**运行在日志里看起来是**有条件**的。

`Preference.normalized` 与 `RewardWeights.with_preference()` 由测试断言**逐点相等**——打分侧和模型侧
共用同一个定义，否则这个旋钮会说谎。参数量代价 **2×4 = 8 个**（<0.01%），不构成容量混淆。

这是「像 RL 但不完全是 RL」的架构落点：**一套策略服务一族权衡点，推理时移动操作点而不重训**。

#### 仍未做

- **训练侧还没采样偏好。** BaRP 训练时在 1-simplex 上均匀采样 `w`；现在模型**能**接受偏好，
  但 `training.py` 还没按批次采样并传入，所以还训不出「一套策略多权衡点」。
- `policy_gradient_loss` 缺熵正则（BaRP 用 β=0.05）。
- 没有 bandit-feedback 训练模式（采样动作、只观测该臂、其余掩码）。
- 基准侧：`implied` 只有 68 组对比对，瓶颈是 `given_text` 对象数量（local 候选 163）。


### 当前轮（2026-10-13）：slot-product 基准被证明是「读标记」，已加 `implied` 模式

原计划是「用 `frame_product_train.jsonl` 重训四臂」。**先做了一步该先做的事：量这个基准能不能区分模型。**
答案是**不能**。

写了一个**只读显式标记、完全不用语义**的分类器（来源前缀、`直接列出结果` vs `并解释原因或给出方案`、
步数子句、风险括号、以及 underspecified 行里直接打印的 `来源线索：device；预计2步；medium风险；extract`），
在基准和训练集上：

| 语料 | marker-only 准确率 | 多数类 | 模型最好成绩 |
|---|---:|---:|---:|
| `frame_product` benchmark (960行) | **1.0000** | 0.2500 | 0.9740 |
| `frame_product` train (2880行) | **1.0000** | 0.2500 | — |
| `hard_benchmark`（对照，138行） | 0.2609 | 0.2609 | 0.4783 |

**1.0000 意味着这个语料是一个可确定性解码的标记查找表**：任何在同一分布上训练的模型都会饱和到
96-97 分（实测正是如此），因此**重训四臂得到的表无法回答编码器问题**。手写基准同一读取器只有
0.2609（等于瞎猜），这才是「难但可判」的区间。

#### 修复：`ProductOptions.naming`（`explicit` | `implied`）

新增 `implied` 模式，**不命名任何 deciding slot**：

- `source` / `risk` 由**对象本身的含义**决定（`_OBJECT_CONTEXT`，38 个对象：`航班状态`→service、
  `用药剂量`→高风险）；
- `derivation` 由**谓语含义**决定（`_PREDICATE_DERIVATION`：`统计`→extract、`分析`→construct）；
- 步数用**多种自然句式**表达，避免单一标记。

| 语料 | frames | rows | marker-only | 四路 |
|---|---:|---:|---:|---|
| explicit benchmark | 320 | 960 | **1.0000** | 240/240/240/240 |
| **implied benchmark** | **96** | **288** | **0.2500** | 72/72/72/72 |
| **implied train** | **288** | **864** | **0.2500** | 216/216/216/216 |

marker-only 降到 **0.2500 = 多数类**，即**没有标记可读了**。

#### 过程中修掉的三个真缺陷（全部由测试抓到）

1. **extractive 帧被静默丢弃**：`steps=1` 时两个手写变体完全相同（alternate 只改步数子句），
   去重检查把它们全部丢掉，于是 **local / rag 候选为 0**，整个语料只剩 cloud 和 review。
2. **38 个对象的 underspecified 行文本全同**（常量句），产生 111 条重复——跨 split 就是泄漏。
   现在措辞里带上对象（"帮我把{对象}那件事处理一下"）但不带操作。
3. `ProductOptions.naming` 未校验，非法值静默走 explicit 分支。

另外 `_generate_cached` 现在把配额**钳制到最小路由的候选数**：语义受限的乘积合法组合本来就少，
`implied` 只有 384 frame 而不是 1280——**这个缩小是设计信息，不是要抹平的缺陷**。

#### 功效分析：终于把「12 行太少」变成了数字

一直写「样本太少」「aux 不显著」但从没算过。按 95% 置信 / 80% 功效的双比例检验：

| 要检测的 collapse 差 | 需要的对数 |
|---|---:|
| 20pt（0.50→0.30） | 91 |
| 20pt（0.30→0.10） | 59 |
| 10pt（0.20→0.10） | 197 |
| 5pt（0.10→0.05） | 432 |
| 3pt（0.06→0.03） | 746 |

反过来看已经跑过的实验：

| 对比 | 实际差 | 需要 | 实有 | 判定 |
|---|---:|---:|---:|---|
| aux vs plain（hard benchmark collapse） | 5.3pt | **837** | 19 | 严重欠功效 |
| aux vs plain（product collapse） | 2.2pt | **1449** | 137 | 严重欠功效 |

**137 对在 5.8% 基线上只能检测 5.7pt 的下降**（即必须降到 0）。所以上一轮 aux 的
「方向一致但不显著」不是措辞保守，而是**仪器只对大幅度效应有效**。这条约束现在写进了测试。

#### 补完：对象 → 允许谓语的白名单（已修）

`_PREDICATE_GROUPS`（10 组动词）+ 每个对象的允许组。分组同时就是 derivation 类别
（读/列/数/取/核对/换算/翻译 = extract；分析/评估/综合 = construct），所以自然语言的选择
和路由规则是同一件事，不是两套表。

- **不再有「预测安装步骤」「统计培训材料」**；`统计/计算` 只出现在可数对象上，有测试钉死。
- 对象表扩到 **55 个**（document 8 / user_data 11 / service 6 / given_text 17 / none 7，
  high-risk 8 个），覆盖四路。
- **候选 962，配额被 local 的 163 卡住** → 652 frame / 1956 行，benchmark 164 frame / 492 行。

| 语料 | benchmark frames | 对比对 | marker-only |
|---|---:|---:|---:|
| explicit | 320 | 137 | **1.0000** |
| **implied** | **164** | **59（全为最小对）** | **0.2500** |

**59 对只能解析 ~20pt 的差异（基线 30% 时）**。要检测 10pt 需要 ~200 对，即 benchmark 要
~550 frame，而瓶颈是 `given_text` 对象的数量（local 候选 163）。**这是下一步要扩的表，
不是要调的阈值。**

---

### 上一轮（2026-10-13）：reward 数据层落地，暂不启动 RL

当前确实已经进入 reward 搭建阶段，但还没有进入 policy-gradient 训练阶段。已有的
`RouterBenchConvertor` 现在把每个模型的质量、成本、延迟、可用性和模型名写入
`Candidate` 反事实表，并按 `utility-v2` 重新计算候选 reward：

```text
quality_weight * quality
- cost_weight * cost / cost_scale
- latency_weight * latency_ms / latency_scale_ms
- error_cost(route) * (1 - quality)
```

- `available=false` 或质量低于阈值的候选不可选，reward 在运行时为 `-inf`；没有任何可用模型时
  自动保留终态 `review` 候选。
- availability 缺失保留为 `null`，`reward_audit` 单独报告 missing，避免把没有观测到的可用性
  当成成功。
- reward 配置写入每行 metadata，`OfflineRouterEnv` 默认复用数据中的配置；显式传入配置才会覆盖它。
- `minijev.convert routerbench --audit` 现在报告可用动作数、不可用数、缺失数、每行可选动作覆盖和
  best-route 分布；这一步是离线 bandit 前的门禁。

本轮仍不训练 RL。当前 fixture 只有 4 行且没有真实 availability，仍是 plumbing fixture；真实
RouterBench 导出和实测人工误路由代价到位后，才进入 LinUCB / 离线 Thompson baseline，再考虑
policy gradient。下一步优先是拿真实导出跑 `--audit`，确认每路动作支持和 reward 分布，而不是先改算法。

### 本轮（2026-10-11）：四领域语料 + 上下文状态 + 槽位辅助监督

按提议做了三件事：训练集按**新闻/哲学/文学/学术**四域划分、给样本加**上下文状态**、
把 **frame 槽位当辅助监督目标**。三件都落地，两件给出**负面但有信息量**的结果。

#### 1. 四领域语料（结构上是对的）

`minijev/domains.py`：每域每段文本配**同样四个任务**，路由由 frame 推导，所以每段文本恰好贡献
四路各一行，从构造上保证 **25/25/25/25**（`domain_audit` 可复算，测试锁死）。

这条约束不是洁癖，是上一轮量出来的：MASSIVE 派生 rubric 里 **cloud = 0**，并进去后 pooled 的
cloud recall 从 0.58 掉到 0.12、transformer 掉到 **0.00**。**缺席的类别会被学成「不存在」**。

#### 2. 上下文状态（机制通了，数据严重不足 —— 负面）

新增 `minijev/contexts.py` + tokenizer **轮次分隔符**（`turn_sep_id=257` / `vocab_size=258`，
写进 sidecar，单轮 checkpoint 仍按 257 加载）。**context pair = 同一句话、不同状态、不同路由**：

```
query:  这两种能不能一起用
ctx A:  我在整理两种清洁剂的成分表。          -> local
ctx B:  我母亲在吃降压药，医生又开了安眠药。   -> review
```

**实测（argmax，6 组留出 pair）**：

| 臂 | collapse | both_correct |
|---|---:|---:|
| plain（无分隔符） | **16.7%** | 16.7% |
| aux | 50.0% | 0.0% |
| aux+context | 50.0% | 0.0% |

**加了 context 训练反而更差。** 训练侧只有 **6 组 / 24 行**，分隔符 id 只在这 24/4201 行出现过，
embedding 基本没训练。另外修正我自己的一个说法：**分隔符不是用上下文的前提**——plain 臂把上下文与
请求直接拼接，上下文文字照样在输入里，所以能蒙对 1/6。探针仍然有效：**6 组里最多对 1 组，
没人真正会用状态**。结论：这条路要几百组样本才能评估。

#### 3. 槽位辅助监督（方向为正，幅度小，未达显著 —— 半负面）

`JevConfig.aux_slots` + 五个头，`--aux-weight` 开启。三条性质测试锁死：标签与 frame 一致、
**决策路径不读这些头**、**不是容量把戏**（780 / 20,868 = **3.7%**）。

| 臂 | 合成test | news | phil | lit | acad | hard acc | **hard collapse** | **product collapse(137对)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plain | 0.578 | 0.778 | **0.833** | 0.917 | 0.667 | 0.4638 | 21.1% | 5.8% |
| **aux** | **0.593** | 0.778 | 0.806 | **0.972** | 0.667 | **0.4783** | **15.8%** | **3.6%** |
| aux+context | 0.588 | 0.750 | 0.833 | 0.944 | 0.583 | 0.4783 | 21.1% | 2.9% |

- **两个留出集的 collapse 同向下降**：hard 21.1%→15.8%、product 5.8%→3.6%；
  `both_correct` 在 product 上 92.0%→96.4%（137 对）。
- 但 hard 上 8 组最小对里 3→2 组，是噪声；product 上 5.8% vs 3.6% 大约差 1 个标准误。
  **所以「试试看」的答案是：方向一致、代价可忽略（<4% 参数、推理路径零改动）、
  但现有基准还证不到显著。** 留作默认关闭的可选项是合理的。
- 注意 product 上**准确率已经饱和**（96-97%），但 **collapse rate 仍然可分辨**——说明这个基准
  该看的是 collapse，不是 accuracy。

#### 4. 顺带发现：`frame_product` 的生成文本**太直白**（构造批评）

生成文本把来源写在句子里——"查**设备里的**安装步骤，直接列出结果，然后核对关键依据"。模型只要
学会"设备里的→local"这种**显式关键词**就能拿 96+。**难例之所以难，正是因为线索不可靠或缺失**，
所以建议生成文本时**不要命名来源**，让来源只能从动作或上下文推出。它的 pair 分析仍然有价值。

#### 5. 修掉的一个真缺陷（frame_product）

差分测试发现 **27 条 query 文本重复**：underspecified 的措辞不含 derivation/steps，6 个组合生成
同一句话；**重复文本跨 train/benchmark 就是泄漏**。修复：underspecified 的 derivation/steps
**语义上本就未定义**，钉到规范值（EXTRACT/1），使文本成为 frame 的单射函数。
现在 **3840 行文本全唯一**，`_candidate_cases` 加断言防回退。

#### 6. 版本控制

`git init` 完成（88 文件，`.git` 221 KiB，checkpoint/原始语料/派生 JSONL 全部 gitignore）。
基线提交 + 本轮改动待提交。

---

### 历史轮（2026-10-11）：槽位叉乘生成已落地

沿用上一轮的结论，先扩充测量仪器，再考虑多任务槽位监督。新增
`minijev.frame_product`，把 `predicate × object × source × derivation × steps × risk × underspecified`
的合法组合生成成可审计的 `RouteFrame`，路由仍由 `frames.route_from_frame` 唯一推导。

- 默认生成 **1,280 frame / 3,840 行**，四路各 320 frame，逐路留出 80 frame 作为 benchmark；
  训练侧 960 frame，benchmark 320 frame，所有变体按 semantic group 同侧。
- 生成 benchmark 自动从留出 frame 中找出 **137 组互不复用的最小对**；`minijev.boundary`
  会自动识别 `slot-product-v1` 并报告这些对的 collapse rate，不再把千级 frame 当成只有总准确率的平面集。
- 产物为 `data/frame_product_train.jsonl`、`data/frame_product_benchmark.jsonl` 和
  `data/frame_product_manifest.json`；manifest 固化槽位、路由和 split 计数。
- 生成器与审计有独立测试，默认产品保证四路平衡、全部 deciding slot 有覆盖、行文本唯一，且
  不会把 benchmark 的语义组泄漏进训练侧。

这一步只扩大边界证据，不改变 pooled 默认编码器，也不把生成 benchmark 自动并入训练命令。
下一步应在固定 seed / policy 下用 `frame_product_train.jsonl` 重训四臂，并用生成 benchmark 与原始
`hard_benchmark.jsonl` 分开报告 collapse rate；多任务槽位监督仍排在这之后。

作为链路烟雾测试，直接拿旧 checkpoint 做零样本 product benchmark 得到 pooled / merged
`acc=0.3125, collapsed=61.3%`，transformer / merged `acc=0.2771, collapsed=31.4%`。
这两个数字只证明新 benchmark 和 pair 分析可运行；训练语料没有覆盖 product frame，不能用来改写
编码器结论，必须先按同一 product train 重训再比较。

### 最新一轮（10-01 夜）：hard 集重建为 frame 标注 + 边界基准 —— **默认编码器仍不换，且证据更强了**

上一轮的问题不是 hard 集「太小」，而是它的**标注方式是错的**：`(query, "rag")` 是一个**断言**，
没说为什么，所以既不能校验、不能变体、也不能争论。这一轮把它换成 **frame**（结构化槽位），
路由由一条明文规则**推导**出来。同时造出 138 行的**留出边界基准**，第一次真正测出了编码器差异。

#### frame 标注：为什么不是主谓宾

「用主谓宾标注」的直觉对（结构化胜过整体判断），但**主谓宾不是路由的判别结构**。看 spec 自己那对：

```
分析产品手册中的安装步骤        -> rag
分析这套系统为什么存在并发问题   -> cloud
```

两句都是 `[谓] 分析 + [宾] 名词短语`，主谓宾完全相同，路由相反。真正决定路由的是
**状语（答案必须从哪来）** 和 **补语（答案是已存在还是需构造）**：

```
来源=外部手册, 答案=已存在(取回)   -> rag
来源=给定文本, 答案=需构造(推理)   -> cloud
```

frame 六个槽位：`predicate` 谓语 / `object` 宾语 / `source` 状语 / `derivation` 补语 /
`steps` 步数 / `risk` 情态 / `underspecified` 省略。**能改变路由的只有其中五个**，它们叫
*deciding slots*。

#### 推导规则（唯一一份，`frames.route_from_frame`）

```
risk HIGH            -> review    # 答错的代价压倒一切
underspecified       -> review    # 无从下手
CONSTRUCT 或 steps>1 -> cloud     # 答案必须被构造
外部来源              -> rag       # 答案必须去取
否则                 -> local     # 答案已在手边
```

**frame bank 里不存路由。** 路由是算出来的，所以 frame 和 route 不可能漂移；改规则即全库重推。
每行还记录 `route_rule`——决定该行的那一行理由，错误可以归因到**槽位**而不是整句话。

#### 边界基准与 collapse rate

`data/hard_benchmark.jsonl`：**46 frame / 138 行**，每路 11–12 frame，训练路径永不读取。
由 **19 组 contrast pair** 组成，每对两半一起留出（只留一半就测不了「分离」）。其中 **8 组是
minimal pair**——只有**一个** deciding slot 不同，所以失败可以直接归因到那个槽位。

`minijev.boundary` 报的**头号指标是 collapse rate**：一对里两半被答成**同一个**路由的比例。
一个模型可以在基准总分上看着不错，却把每一对都塌掉——这正是基准存在的意义。

#### 对照结果（argmax，隔离 ladder 影响）

| 臂 | 合成 test | 文学 test | MASSIVE | **边界 acc** | **both_correct** | **collapsed** | minimal collapsed |
|---|---:|---:|---:|---:|---:|---:|---:|
| pooled / merged | 0.533 | 0.889 | 0.207 | **0.522** | **31.6%** | **15.8%** | **12.5%** |
| **transformer / merged** | **0.623** | 0.889 | **0.272** | 0.515 | 21.1% | **36.8%** | **25.0%** |
| pooled / +massive | 0.392 | 0.861 | 0.788 | 0.312 | 0.0% | 68.4% | 62.5% |
| transformer / +massive | 0.523 | **0.972** | **0.802** | 0.254 | 10.5% | 68.4% | 50.0% |

多数类基线 = 36/138 = **0.261**。

**结论（三条，都推翻或修正了上一轮的判断）：**

1. **Transformer 在边界难例上没有稳定超过 pooled —— 反而更差。** 总分 0.515 vs 0.522 是平手，
   但 **collapse rate 36.8% vs 15.8%，是两倍**；minimal pair 上 25.0% vs 12.5%，也是两倍。
   按既定判据（「只有 Transformer 在 hard cases 上稳定超过 pooled 才换默认编码器」）
   **判据不满足，而且输在了判据最在意的那一项上**。它在合成测试集上的优势（0.623 vs 0.533）
   是真的，但没有转化为边界判别力。
2. **把 MASSIVE 并进训练会摧毁云路由。** MASSIVE 派生 rubric 里 **cloud = 0 样本**；并进去之后
   pooled 的 cloud recall 从 0.58 掉到 **0.12**、transformer 掉到 **0.00**，`steps 2+` 槽位准确率
   归零。**「合并所有数据」是错的**——一个缺席的类别会被模型学成「不存在」。文学域上 Transformer
   反而保住了（0.972 vs 0.861），说明它对分布偏移更稳，但那不能弥补边界上的损失。
3. **两者都还有巨大空间。** 0.52 是多数类 0.261 的两倍，说明基准不是玄学；但离规则允许的 1.0
   差得很远。逐槽位看，pooled 的 `source_class` external=0.67 / internal=0.52，transformer 的
   `underspecified` True=1.00 但 `source_class` external=0.47——**两个编码器各修好了一半槽位，
   没人修好全部**。

#### 本轮修掉的两个数据缺陷

| 缺陷 | 怎么发现的 | 修复 |
|---|---|---|
| 2 行基准 query **逐字出现在训练集里**（`帮我处理一下` / `这个怎么办`，与模板库 `review:t13/t14` 撞词） | 写 `assert_no_text_overlap` 做全文比对 | 重写 8 条 underspecified frame 的措辞；把碰撞检查固化为测试 |
| `VARIANT_SUFFIXES` 声称第三个变体是尾部改写，代码却从未用到那个后缀 | 逐行核对 `_texts` 与文档 | 后缀改为真实生效，并加测试钉死「第三变体是尾部变体，不是独立措辞」，同时报 `hand_written_texts` 防止虚增样本量 |

#### 仍然存在的改进空间（用户直接问的）

- **训练侧边界数据太薄**：30 frame / 90 行，benchmark 是 46 frame 的**未见过** frame。
  pooled/transformer 都只有 ~0.52，一半的错误来自「没见过足够的边界样本」，不全是编码器问题。
- **76 个 frame 还是手写的。** 下一步应当是**槽位叉乘生成**（predicate × object × source ×
  derivation × steps 的合法组合），把库从 76 扩到 1000+，这是唯一能把 collapse rate 的
  置信区间压下来的办法。
- **最强的一个想法：把 frame 槽位当辅助监督目标。** 既然路由是槽位的函数，直接让模型多任务预测
  `source/derivation/steps/risk` 可能比从表面文本硬学路由更省样本。这一轮没做，但它现在是
  有依据的下一步——而且它是**用上了新标注本身**的唯一方案。
- 基准是合成构造的，作者既是标注者又是被评测者；换成真人标注前，所有绝对值只能内部比较。

---

## 历史状态（10-01 晚）：Transformer 对照实验

### 结论：默认编码器不换

本轮验收了 Transformer 编码器、本地文本生成、路由误判代价三项新增，补了 23 个测试（**138 全绿**），
并按「同数据 / 同 seed / 同 policy / 同 tokenizer」跑了 5 臂对照。**结论：Transformer 在合成与文学域
大幅领先，但在 spec §6.1 边界难例上没有超过 pooled —— 按既定判据，默认编码器保持 pooled。**

#### 对照结果（held-out 测试集，argmax policy 以隔离 ladder 影响）

语料 = 合成+文学（911 行训练）；val 为合成 val + 文学 val（149 行），四条路齐全。

| 臂 | 参数量 | 合成 test(140) | 文学 test(36) | hard cases(12) | MASSIVE(2974) |
|---|---:|---:|---:|---:|---:|
| pooled | 20,868 | 0.486 / mF1 0.472 | 0.972 | **8/12 = 0.667** | 0.218 |
| **pooled-640（参数量对齐对照）** | 60,612 | 0.493 / 0.488 | 0.972 | 8/12 = 0.667 | 0.195 |
| **transformer** | 62,532 | **0.536 / 0.511** | **1.000** | **8/12 = 0.667** | **0.286** |

语料 = 合成+文学+MASSIVE（12,425 行训练）；val 2,182 行。

| 臂 | 合成 test | 文学 test | hard cases(12) | MASSIVE |
|---|---:|---:|---:|---:|
| pooled | 0.493 / 0.345 | 0.528 | **6/12 = 0.500** | 0.791 |
| transformer | **0.693 / 0.541** | **0.944** | 5/12 = 0.417 | **0.807** |

#### 三条结论

1. **Transformer 的优势来自注意力，不是参数量。** 把 pooled 加宽到 60,612 参数（与 Transformer 的
   62,532 相差 3%）后，val macro-F1 反而从 0.4669 掉到 0.4520；Transformer 是 0.6906。**参数对齐
   对照把「只是容量大」这个解释排除掉了。**
2. **混合语料下 Transformer 的鲁棒性差距极大。** 加入 MASSIVE（11.5k 行）后 pooled 的文学域从
   0.972 崩到 0.528，Transformer 只从 1.000 降到 0.944。看着像 pooled 被大语料拖去拟合表面形式，
   而注意力能保住长提示末尾的任务线索。
3. **但边界难例上 Transformer 没有赢：** 合成+文学语料上两者都是 8/12；加入 MASSIVE 后 pooled
   6/12、Transformer 5/12。**按「只有在 hard cases 上稳定超过 pooled 才换默认编码器」的判据，
   这一条不满足。**

#### 一个必须先修的方法论问题

**12 条 hard cases 当不了判据。** 一行翻转就是 8.3 个百分点，8/12 与 5/12 的差别在二项噪声内，
「稳定超过」用现在的仪器根本测不出来。要电气化这个判据，先得把 hard 测试集扩到 **60–100 条以上**
（现有 28 组难例里只有 6 组落在测试集），或者改变划分比例单独保留一批难例测试集。**在此之前
编码器之争应当挂起**，而不是用 12 行数据定夺。

#### 置信度校准

| 臂 | 合成 test ECE | 文学 ECE | MASSIVE ECE | 文学 conf spread |
|---|---:|---:|---:|---:|
| pooled | 0.305 | 0.197 | 0.531 | 0.778 |
| transformer | 0.429 | **0.015** | 0.637 | **0.145** |

Transformer 在它满分的文学域上 ECE 只有 0.015，但 confidence spread 掉到 0.145 —— 路由分布接近
one-hot，ladder 几乎没有可用的不确定度（升级率 0%）。**「全对且从不犹豫」对准确率是好事，对安全网
是坏事**，这一点要在选编码器时一并考虑。

#### 本轮验收与新增

| 项 | 结果 |
|---|---|
| Transformer 编码器 | forward / 掩码 / 存读 / 配置校验全部通过；**投影层已确认为持久参数**（state dict 含 `transformer_projection.weight`，重复 forward 逐位相同） |
| 存读一致性 | 权重逐位相同；logits 差异 4.5e-08，是 kernel 归约顺序，不是权重丢失 |
| 本地文本生成 | `generate_from_text` 分块、块索引、变体分组、provenance 全部验证（用临时文本，未编造小说内容） |
| 路由误判代价 | `error_costs` 默认空、按 `(1-quality)` 缩放、字符串键、可比翻 reward 排序，均已测 |
| 测试 | **115 → 138**（新增 `test_model_encoders.py`、`test_literary_text.py`，`test_rewards.py` 补 6 条） |
| JIT | 开启后 Transformer 从 61.6 s/epoch 降到 13.6 s/epoch（4.5×），实验因此可负担；两臂同设置，公平 |

> ⚠️ **《局外人》文件在本机不存在。** 我搜了 Downloads / Desktop / Documents / D:\ / E:\ 全盘，
> 没有任何匹配 `局外人|stranger|outsider|camus|加缪` 或小说体量的文本文件。本轮实验因此用的是内置
> 12 段古文补充集。拿到文件后重跑 `python -m minijev.literary --text-file ...` 即可复现文学域对照。

---

## 历史状态（10-01：RouterBench 与文学数据接入）

最新一轮（10-01）：RouterBench 转换器与文学数据已接入并审计；修复了两个缺陷（reward 反事实表
缺安全维度、文学生成器变体重复）；合成数据补入 28 组 spec §6.1 边界难例；跑完 3 模型 × 3 测试集的
跨域矩阵。**115 个测试全绿。**

### 本轮的关键测量（3 模型 × 3 测试集，含 ladder 升级）

| 模型 | 合成 test（20 组） | 文学 test（12 组） | MASSIVE test（2,974） |
|---|---:|---:|---:|
| synthetic+hard | 0.393 / 0.391（argmax 0.493） | 0.306 | 0.182 |
| literary-only | 0.050 | **0.972** | 0.080 |
| merged（合成+文学） | **0.450** / 0.462 | **0.972** | 0.202 |
| 多数类基线（训练集拟合） | 0.186 | 0.250 | 0.497 |

三个结论：

1. **域迁移不发生。** 合成模型在文学测试上 0.306（文学模型自己 0.972）；字节级 mean-pool 学的是
   每个域的**表面形式**。每进一个新域都要补该域的数据，然后**合并训练**（merged 在合成上反而
   超过 synthetic-only：0.450 vs 0.393，文学上不掉：0.972）。
2. **边界难例把模型的真实弱点钉死了。** 12 条 hard cases 上 acc 0.583；spec 自己给的例子
   「分析产品手册中的安装步骤 → rag」被模型的「分析 → cloud」表面联想带走。这正是 §6.1 设计这些
   样本的目的——现在失败被**证明**了，Transformer/更富特征才有资格进场（README 第 6 条）。
3. **派生 confidence 在域内有效、域外偏自信。** MASSIVE 上 spread 0.9966 但 acc 只有 0.18-0.20：
   分布锐利 ≠ 正确。升级率在域外自动升高（44%），安全网有反应，但方向判断还要靠标签。

### 本轮修复的两个缺陷

| 缺陷 | 修复 | 测试 |
|---|---|---|
| RouterBench 反事实表无安全维度：医疗行（全部模型不达标）的 reward 表会选最大模型，RL 会学到把医疗路由给 cloud | `Candidate.eligible` 字段；不达标候选在 reward 表中记 -inf；无候选达标时合成终态 review 候选（带声明的审核成本/延迟先验） | `test_convert.py` 4 条 + `test_rewards.py` 2 条 |
| `literary.py` 的 variant 1/2 生成相同文本（144 行只有 96 个不同文本） | 每变体一个不同的措辞后缀，测试钉死「48 组 × 3 个不同措辞」 | `test_literary.py` |

RouterBench 原始数据（HF `withmartian/routerbench`）本机不可达，fixture（4 行，四路各一）走通全链路；
拿到真实导出后按 `data/external/routerbench/README.md` 用 0.7/0.8/0.9 三个阈值做审计再组训练集。

---

## 历史状态（2026-09-29：阶段 A/B/C.1 已执行 + confidence 已对齐 TypeSafe 契约）

`§7 阶段 A`、`§7 阶段 B`、`§7 阶段 C.1` 已完成。`§3.2` 的模块划分落地为 16 个模块 + 99 个测试。

### 已落地的结果

| 项目 | 重构前 | 现在 |
|---|---|---|
| 合成数据划分 | 逐行随机，**117/129** 个测试 query 与训练重叠 | 按语义组划分，**0** 泄漏（`assert_no_group_leak` 每次执行） |
| 合成数据测试集 | Acc **0.9808**（模板记忆） | Acc **0.6106** / Macro-F1 0.5916（未见模板，含 ladder 升级 10.6%） |
| MASSIVE 测试集 | Acc **0.2145**（v1 rubric，零样本） | Acc **0.7683**；三类口径 Macro-F1 **0.6117** |
| 规则基线 vs 多数类基线 | 只有规则基线 | 多数类基线常驻报告（且在训练集上拟合，不在测试集上） |
| 推理链路 | `examples/infer.py` **永远跑随机权重** | `minijev.inference` 默认加载 checkpoint，缺失时抛 `CheckpointMissing` |
| confidence | 独立回归头，spread **0.0175**，`collapsed=True` | 从 route 分布派生，spread **0.9133**，`collapsed=False` |
| 测试 | 0 个 | **99 个**，`ruff` 全绿 |
| 模块 | 根脚本 4 个 + `data/` Python 包 + 3 个空目录 | `minijev/` 16 个模块，三个入口共享核心 |

### confidence 的架构改动（对齐 TypeSafe 契约）

原设计有一个独立的 confidence 回归头，预测数据集的「标注一致性」标签。这是个**类别错误**：
一个被训练去回归数据集标签的头，无法知道**这一次预测对不对**。标签 96.5% 是同一个值，头就学成了常数。

| 信号 | 226 行测试集上的 spread |
|---|---:|
| `max(route_probs)` | 0.642 |
| `1 − H/log(4)`（默认） | **0.944** |
| 被删掉的 confidence 头 | **0.018** |

分布本来就带着信号，头把它盖成了常数。派生指标还能区分对错：**答对时平均 0.700，答错时 0.445**。

改动：删掉这个头（模型从 20,933 参数变 20,868）；`confidence` 改为 `minijev.confidence` 从
`route_probs` 派生的统计量，提供 4 种可插拔度量；`JevSample.confidence` 保留为**训练样本权重**
（`--no-agreement-weights` 可关），不再作为预测目标。训练日志里 confidence 从 0.003 升到 0.770、
升级率从 0.971 降到 0.000——模型学会之后安全网自己停止触发。

### 分路阈值（TypeSafe 的「阈值随风险变化」）

从单一 `review_threshold=0.35` 改为 `RiskLadderPolicy`：每个 route 有自己的 confidence 门槛和升级目标。

**关键修正**：第一版把 `local` 的门槛设得最高（0.50），这是**反的**——它把最便宜、最可恢复的动作
当成了最需要信心的动作。按风险重排后（local 最松、cloud 最紧），MASSIVE 上实测：

| ladder | accuracy | macro-F1 | 升级率 |
|---|---:|---:|---:|
| 不升级（`argmax`） | 0.7724 | 0.5585 | 0.0% |
| **当前默认** | 0.7683 | **0.6117** | 3.1% |
| 第一版（local 最严） | 0.6809 | 0.5903 | 22.1% |

代价 0.4 个准确率点，换 5.3 个 Macro-F1 点，只升级 3.1% 流量。

**但要诚实说**：在两个数据集上，per-route ladder 相对 flat threshold（0.35）**没有明显优势**
（MASSIVE：0.6117 vs 0.6157）。因为目前没有任何**错误代价模型**——四路的错的代价被当成一样。
ladder 的价值是结构性的（能表达不对称代价），要等能说出「cloud 误判比 local 误判贵多少」时才兑现。

### 仍未解决的问题（下一阶段）

1. **测试集太薄。** 77 个语义组三分之后只剩 15 个组进测试；诚实划分的代价就是测试集在**内容**上很小。
   需要把模板数扩到几百个量级，或引入真实数据。
2. **ladder 门槛是推理出来的，不是拟合出来的。** `minijev.evaluation --sweep` 给出曲线，
   但要选点必须先有误路由代价，那是产品决策。
3. **`review` 仍然学不动。** MASSIVE 上 F1 只有 0.249（169 条，只有一个意图），合成数据上 0.593。
   需要真实的 review 来源。


---

## 0. 本轮新增的实测证据

为了判断 MASSIVE 上的 21.45% 到底是「模型不行」还是「映射不对」，我做了**单变量对照实验**：用**完全相同的架构、相同的 mapping、相同的超参**，只把训练数据从 315 条玩具模板换成 MASSIVE 自己的 train split。

```powershell
python data\external\massive\convert_massive.py --data ...\zh-CN.jsonl --output-dir data\external\massive\derived
python train_dataset.py --train ...\massive_train.jsonl --validation ...\massive_val.jsonl `
  --test ...\massive_test.jsonl --epochs 20 --batch-size 128 --patience 5 `
  --output models\massive_sft.safetensors --metrics models\massive_sft_metrics.json
```

同一份 MASSIVE zh-CN 官方 test（2,974 条），三种状态对比：

| 状态 | Accuracy | Macro-F1 |
|---|---:|---:|
| 零样本（`minijev_phase1` 权重） | 0.2145 | 0.1782 |
| 多数类基线（全预测 local） | 0.4694 | — |
| **微调后（同架构，只换训练数据）** | **0.7915** | **0.5975** |

**结论：架构、字节 tokenizer、tinygrad 训练链路都没有问题。21.45% 的唯一原因是它只见过 315 条玩具模板。** 参数量仍是 20,933，训练 508.8 秒（best epoch 19/20）。

微调后的分路表现反过来说明了很多事：

| 派生 route | Recall | Precision | F1 | 测试集占比 |
|---|---:|---:|---:|---:|
| local | 0.848 | 0.810 | 0.829 | 46.9% |
| rag | 0.837 | 0.794 | 0.815 | 44.2% |
| cloud | 0.521 | 0.576 | 0.547 | 3.2% |
| review | 0.124 | 0.500 | 0.199 | 5.7% |

**语义连贯的两个类（local/rag，占 91% 数据）拿到 ~84%；语义是硬凑的两个类（cloud 只有 3 个 recommendation 意图、review 只有 `general_quirky` 一个意图）分别只有 52% 和 12%。** 模型学到的是 mapping 的真实含义——哪里 mapping 讲得通，哪里就学得好。

> ⚠️ 一个不能当结论的数字：微调后 `confidence_mae = 0.0012`。这是因为我给所有 MASSIVE 样本的 confidence 标签都填了 1.0（MASSIVE 没有 confidence 标注），confidence 头退化成了常数。**这次实验只验证了 route 头，没有验证 confidence 头。** 见 §1.3。

---

## 1. MASSIVE 结果的正确解读

### 1.1 先确认一个坏消息：confidence 头确实是个常数

我在上一版代码地图里的 P0-2 预判，在真实数据上被证实了：

| | 数值 |
|---|---|
| 训练集 confidence 标签均值（793×1.0 + 243×0.67） | **0.9226** |
| MASSIVE 上四个 route 的平均 confidence 输出 | **0.9147 / 0.9163 / 0.9174 / 0.9172** |
| 四者跨度 | **0.00276** |

模型在一个完全不同的数据集上，输出的 confidence 几乎精确复现了训练集的 base rate。**它对输入不敏感——这不是「不确定」，是「没学会」。** 连带后果：`router.py:15` 的 `confidence < 0.35 → review` 覆盖在 MASSIVE 上同样永不触发。

### 1.2 一个需要纠正的判断：21.45% 不能用来证明「映射有问题」

「先审计 MASSIVE 到四路 route 的映射」这个下一步方向是对的，但**理由不是 21.45%**。原因是：

```python
# evaluate_massive.py:88
def derived_route(intent: str) -> Route:   # route 是 intent 的纯函数
```

**route 是 intent 的确定性函数**，所以这套标签**在构造上 100% 自洽，零标签噪声**。准确率的上限只取决于模型能不能提取出判别 intent 的特征，与 mapping 是否合理无关。上面的微调实验直接证实了这一点：同一套 mapping，79.15%。

所以正确的表述是：**21.45% 衡量的是域迁移失败（玩具模板 → 真实口语），不是映射质量。**

### 1.3 映射真正的问题在别处：语义效度 + 类别失衡

审计 mapping 仍然值得做，但要审的是这两件事：

**(a) 类别严重失衡，违反项目自己的规范。** 派生标签分布：

| 派生 route | 占比 | 来源 |
|---|---:|---|
| local | 46.9% | 33 个意图（闹钟/IoT/音乐/播放……） |
| rag | 44.2% | 23 个意图 |
| **cloud** | **3.2%** | **只有 3 个 `recommendation_*` 意图** |
| **review** | **5.7%** | **只有 1 个 `general_quirky` 意图** |

`MiniJev_Dataset_Spec.md:150-159` 明文要求四类各 25%，并警告「不要让 local 占绝大多数，否则模型可以通过预测多数类获得虚假的高准确率」。当前 mapping 正好踩了这条。

**(b) 部分映射在语义上站不住。** 最明显的几个：

| intent | 样本 | 当前映射 | 问题 |
|---|---:|---|---|
| `email_sendemail` | 114 | rag | 发邮件**不需要检索** |
| `social_post` | 81 | rag | 发帖同样不需要检索 |
| `lists_remove` / `lists_createoradd` | 91 | rag | 这是**写操作**，不是检索 |
| `recommendation_movies` | 20 | cloud | 偏好查找而已，够不上「复杂推理」 |

根因是把「设备动作 vs 检索」和「需不需要复杂推理」混成了一个维度。

### 1.4 最重要的结构性问题：MASSIVE 测不了 cloud 和 review

MASSIVE 是**智能音箱指令数据集**（场景分布：calendar / play / qa / email / iot / general / weather / transport / lists / news / recommendation / datetime / social / alarm / music / audio / takeaway / cooking）。它**没有复杂推理意图，也没有高风险意图**。

因此 cloud（3.2%）和 review（5.7%）这两个类是从极少数意图里**发明**出来的。任何基于 MASSIVE 的「四路路由准确率」都无法支撑对外结论。

**MASSIVE 的正确用途有两个，都不是「四路评测」：**
1. **域迁移压力测试**——真实的、口语化的中文用户表达，正是合成模板完全缺失的；
2. **local ↔ rag 边界的真实语料来源**（这两类占 91%，且语义连贯，微调后 84% 是有意义的数字）。

cloud / review 必须另找来源（`Open_Source_Datasets_Survey.md` 已列出：HotpotQA / FRAMES 对 cloud，WildGuard / CLUE 对 review）。

---

## 2. RL 该不该现在做

### 2.1 一个必须先纠正的事实：MiniJev 的直系论文不是 RL 论文

调研核实了两篇源论文的实际形式化：

| | arXiv:2503.23303《SalesRLAgent》 | arXiv:2510.01237《置信度感知路由》 |
|---|---|---|
| 范式 | **RL**（policy net + value net + 元学习） | **纯监督**，没有 RL |
| 路由方式 | RL 学策略 | **确定性阈值规则** `A(C)` → local/rag/large/human |
| 阈值 | — | θ = 0.75 / 0.55 / 0.35 |
| 置信度 | — | `C = w1·C_sem + w2·C_conv + w3·C_learned`，权重由验证集学得 |
| 训练数据 | GPT-4O 合成 120 万条对话 | **仅 72 个样本**，30 epochs |
| 训练流程 | **两阶段**：先监督建立策略初始化 → 再 RL | 单阶段监督 |

**关键推论**：MiniJev 现在的四路 + confidence 头是**第二篇的直译，而第二篇根本没有 RL**。导师说的「RL 是 Jev 的一个阶段」只在第一篇里成立——而第一篇的主题是**销售转化预测**，MiniJev 一行都没实现。

这不是说 RL 不该做，而是说：**在 MiniJev 上做 RL，等于做一件两篇论文都没做过的事。** 这恰好就是「make difference」的机会，但必须说清楚它没有现成的论文背书。

另外三条风险提示（关于文献可信度）：
- 两篇论文**同一作者**（Nandakishor M）、**单作者**、**无代码、无同行评审**；
- 第二篇的结论建立在一个 **72 样本**的实验上；
- 第一篇**从未给出具体 RL 算法名和 reward 公式**，只说 "specialized RL algorithm"。

**所以不要把这些论文的数字当对标目标。** MiniJev 目前 1,036 条数据已经是第二篇的 14 倍。

### 2.2 数学上：路由是 contextual bandit，不是 full RL

调研（routing survey, [arXiv:2603.04445](https://arxiv.org/abs/2603.04445)）明确区分：

- **routing = 单次决策**，query → 一个模型，reward 观测一次即 episode 结束，**没有状态转移**；
- **cascading 才是多步**（先小模型、失败再升级），那才需要 MDP / POMDP。

MiniJev 的四选一是标准 **contextual bandit**。这意味着：

- 「RL 微调」在这里 = 单步 policy gradient ≈ REINFORCE，实现量大约几十行，**成本极低**；
- LinUCB / Thompson sampling 这类 bandit 算法本质就是**带显式探索项的 policy gradient**（MetaLLM 的 LinUCB 选择式 `argmax_j xᵀθ_j + α√(xᵀA_j⁻¹x)`：第一项 exploit、第二项 explore）；
- 所以「用 bandit 还是用 policy gradient」是工程选择，不是范式之争。**bandit 额外送你 regret 界，PG 额外送你能训大网络。**

### 2.3 真正的阻断点：数据里没有 reward，也没有反事实

这是导师说的「数据只有 query/route/confidence，还没做分离」的精确版本：

| RL 需要 | 当前数据 |
|---|---|
| state (context) | ✅ `query` |
| action | ✅ `route` |
| **reward** | ❌ **完全没有** |
| **反事实分支**（其他三个 route 会拿多少 reward） | ❌ **完全没有** |
| propensity（behavior policy 选该 action 的概率） | ❌ 没有（离线 OPE 必需） |

**每条样本只有一个 route 标签 = 单标签监督 = 无法做反事实评估。** 你选了 local，就永远不知道 rag/cloud/review 会得多少分。**没有反事实就没有 RL**，这不是算法问题，是数据契约问题。

### 2.4 好消息：不需要真实后端也能拿到 reward

**RouterBench 就是一个标准的离线 contextual bandit 数据集。** 它的 schema（读源码确认）是宽表：每行一个 prompt，每个模型两列 `"{model}|model_response"` 与 `"{model}|total_cost"`，外加每个模型的 performance 列，还直接提供 `generate_oracle_routing_results()` 产出 `oracle_model_to_route_to`。

即 **(context, action, reward) 对每个 (prompt, model) 组合穷举标注**，405k prompts × 11 模型。离线 reward 标准写法：

```
r(x, i) = a_i(x) − p · c_i        a=正确性∈{0,1}, c=成本, p=cost scaling
```

许可证 **MIT**，最干净。这也是 `Open_Source_Datasets_Survey.md` 推荐的第一优先级数据集。

### 2.5 结论：现在不做 RL，但把接口留对

**收益（低）**：决策空间只有 4 维、监督信号已经足够稠密，1000 条数据上 RL 相对监督分类器预期没有明显提升。

**风险（高）**，四个具体失败模式：

| 风险 | 证据 |
|---|---|
| **路由坍缩** | PickLLM 论文自己报告 "converges to a single LLM for the remaining session queries"。4 个 action + 1000 条样本，坍缩几乎必然 |
| **Reward hacking** | Router-R1 专门用分层 reward（format 不合法则 outcome/cost 清零）对抗，MiniJev 没有任何校验机制 |
| **Router staleness** | 没有真实后端 = reward 是合成/启发式的，学到的 policy 迁移到真实后端会失效 |
| **错误代价不对称** | cloud 选错 = 白花钱，local 选错 = 用户拿到幻觉。单一标量 reward 表达不了 |

**推荐路径**：`SFT 分类器 → 离线 contextual bandit（LinUCB / 离线 Thompson sampling）→ 数据攒够再考虑 policy gradient`。**不要直接上 PPO/GRPO**——那需要 critic、reference model、rollout engine，而 GRPO 至少能省掉 critic，是这个方向上最省的形态。

### 2.6 具体怎么「预留」

**(a) 数据 schema 扩成反事实表（最高优先级，改动小收益大）**

```jsonc
{
  "query": "...",
  "route": "local",            // 保留，向后兼容，仍是监督标签
  "confidence": 0.72,
  "candidates": [              // 新增：4 条，穷举 counterfactual
    {"route": "local",  "reward": 0.8, "quality": 0.8, "cost": 0.0, "latency_ms": 120,   "propensity": 0.4},
    {"route": "rag",    "reward": 0.7, "quality": 0.8, "cost": 0.1, "latency_ms": 400,   "propensity": 0.3},
    {"route": "cloud",  "reward": 0.6, "quality": 0.9, "cost": 1.0, "latency_ms": 900,   "propensity": 0.2},
    {"route": "review", "reward": 0.5, "quality": 1.0, "cost": 2.0, "latency_ms": 3.6e6, "propensity": 0.1}
  ],
  "reward_version": "heuristic-v1",
  "eligible": true,
  "policy_version": "sft-0001",
  "timestamp": "2026-09-29T..."
}
```

**(b) policy 头输出要对 RL 友好**——最容易被做错的一点：

- `route_logits: [4]`，**不是 argmax 后的 one-hot**。RL 需要对 logits 采样并算 log-prob。现在 `model.py` 已经吐 logits，**这一点现有代码是对的，别改坏**。
- 保留 `confidence`。
- 可选预留 `value: scalar`（将来 critic 用，现在只留字段不实现）。

**(c) reward 与训练解耦**（抄 verl 的 `RewardManager` 抽象：把「reward 从哪来」和「怎么训练」分开）：

```python
class RouterEnv(Protocol):
    def step(self, query: str, route: str) -> tuple[float, dict]: ...

class MockRouterEnv:   # 现在：查 candidates 表
class LiveRouterEnv:   # 将来：真调后端
```

**(d) 观测钩子 ≠ 决策钩子**（抄 LiteLLM）：把「给 query 打分」和「观测结果」拆成两个函数。**观测钩子就是将来 RL reward 的天然接入口**，不动推理路径就能接上。

---

## 3. 模块划分：一个反直觉的调研结论

调研了 7 个开源实现（RouteLLM、RouterBench、semantic-router、LiteLLM、RouterEval、LLM-Blender、FrugalGPT），最重要的发现是：

> **这 7 个仓库，没有一个有顶层 `training/` + `inference/` 双目录树。**

训练、推理、评估是**共享同一批核心模块的三个 CLI 入口**，不是三棵复制代码的树：

| 项目 | 训练 | 推理 | 评估 |
|---|---|---|---|
| RouteLLM | 各 router 目录内脚本 | `controller.py` / `openai_server.py` | `evals/evaluate.py` |
| RouterBench | `convert_data.py` | — | `evaluate_routers.py` → `visualize_results.py` |
| LLM-Blender | 根目录 `train_ranker.py` | 可 pip 安装的 `llm_blender` 包 | 独立 `gpt_eval/` 模块 |
| FrugalGPT | 无 | `src/service` | `examples/` + `analysis/` |

**所以「分别实现 training 和 inference 的 pipeline」应该落实为两个入口文件，而不是两个目录。** 两个目录会立刻产生重复代码，而共享核心模块才让「先监督、后 RL」只换 loss 不重写。

### 3.1 反复出现、值得抄的 5 个模式

1. **「产生分数」与「用分数决策」分离**——RouteLLM 的 router 只输出一个标量，阈值逻辑写在基类 `route()` 里，于是阈值可离线校准而完全不动模型。
2. **canonical schema + 独立 convert 阶段**——没人直接在原始日志上训练。`convert_massive.py` 已经无意中走了这条路，应该正式化。
3. **一策略一文件 + 注册表**——RouteLLM 的 `ROUTER_CLS` dict；契约永远是「一个方法 + 固定签名」。
4. **观测钩子 ≠ 决策钩子**（LiteLLM）。
5. **评估产物是曲线/长表，不是一个 accuracy 数字**——RouteLLM 扫阈值画 accuracy-vs-cost 曲线，FrugalGPT 把成本-质量 tradeoff 曲线当交付物。**这正是 MiniJev 该产出的东西。**

### 3.2 建议的最小模块划分（12 个）

```text
minijev/
  __init__.py     只导出 Router facade 与 Decision，别的不外露
  schema.py       唯一数据契约：QueryRecord / DatasetRow / Decision / Candidate / Reward
  config.py       四路 route 定义表、阈值、维度、路径（纯数据）
  data.py         读/写/切分，只认 schema
  convert.py      异构原始数据 -> DatasetRow（MASSIVE / RouterBench 各一个 convertor）
  features.py     query -> 特征；可冻结可缓存
  model.py        tinygrad 网络（纯计算，不认识 local/rag/cloud/review，只吐标量+logits）
  objectives.py   监督损失 + RL 损失接口（reward -> loss）← 唯一的 RL 接入点
  policies.py     score -> route：阈值/成本约束/校准
  routers.py      Router ABC + 注册表（rule / linear / mlp 各一个文件）
  inference.py    推理入口：CLI + predict()，加载 checkpoint
  training.py     训练入口：CLI + fit()，checkpoint 落盘
tests/            test_schema.py  test_policies.py  test_train_smoke.py
```

**三条硬边界**（违反任何一条都会导致后面重写）：

1. `model.py` 不认识四路语义，只输出标量 + logits；
2. 路由决策**只**在 `policies.py`；
3. `training.py` 与 `inference.py` 都只依赖 `data`/`features`/`model`/`policies`，**互相不 import**。

**现有目录怎么处置**：`training/`、`inference/`、`utils/` 是空目录，直接删掉；`train_dataset.py` 收进 `minijev/training.py`，`examples/infer.py` 收进 `minijev/inference.py`，`evaluate_baseline.py` / `evaluate_massive.py` 变成 `evaluation.py` 的子命令。

### 3.3 明确不要抄（小项目学这些会死）

LiteLLM 全套（Redis/DualCache 计数、41KB `budget_limiter.py`、插件系统、686KB 单类、3MB 价格表、Helm/Terraform/UI）、semantic-router 的 `HybridRouteLayer` 与 async 镜像、RouterBench 的 MongoDB 缓存、LLM-Blender 的 DeepSpeed `zero_configs/`。

**通用红线**：不要插件系统、不要 proxy/server、不要 Redis、不要在出现同步瓶颈前写 async 变体、不要把 39MB 的数据集提交进 git。

---

## 4. 修订后的路线图

原则：**先把尺子做准，再拆模块，再换数据，最后才碰 RL。** 每一阶段都能独立验收。

### 阶段 A：让数字诚实（最小化，~1 天）

比上一版更克制——先不做测试框架，只做能让数字可信的最小集合。

| # | 任务 | 验收 |
|---|---|---|
| A.1 | `git init` + 首次提交；`.gitignore` 排除 `data/external/`（那个 MASSIVE tar.gz 39MB）与 `__pycache__` | 有可回滚历史，仓库不含大二进制 |
| A.2 | 给 `generate_dataset.py` 的**全部 1,036 条**样本分配 `semantic_group_id`（现在只有 36 条同义改写有） | 每行都有 group id |
| A.3 | `data/split.py` 改成**按组划分** | train ∩ test 的 group = 0，断言写进代码 |
| A.4 | 重跑 `train_dataset.py`，报告里**同时**给出「全体」和「未见语义组」两个数字 | 出现两个数字，且后者明显更低 |

**预期**：诚实数字会从 98% 掉到 80% 附近。这是好事。

### 阶段 B：拆模块 + 接上推理（~1-2 天）

| # | 任务 | 验收 |
|---|---|---|
| B.1 | 按 §3.2 重排；删掉三个空目录 | 12 个模块到位 |
| B.2 | `inference.py` **加载 checkpoint**（现在 `examples/infer.py` 永远跑随机权重） | 同一 query 的输出与训练后评估一致 |
| B.3 | 评估产出**阈值扫描曲线**（confidence 阈值 vs 覆盖率 vs 准确率），不是一个数字 | 有曲线文件 |
| B.4 | 3 个冒烟测试 | `pytest` 全绿 |

### 阶段 C：换数据（~3-7 天）

| # | 任务 |
|---|---|
| C.1 | 修正 MASSIVE mapping 的**语义**（把 `email_sendemail` / `social_post` / `lists_*` 这类写操作从 rag 移出），并在 README 写明「MASSIVE 只用于 local↔rag + 域迁移压力测试」 |
| C.2 | 引入 **RouterBench（MIT）** 做 cloud 侧的真实监督信号，用它的 (prompt × model) quality/cost 派生四路标签 |
| C.3 | cloud / review 另找来源（HotpotQA / FRAMES / WildGuard / CLUE），**注意 FalseReject 是 CC-BY-NC 禁商用**，与 vision 的开源+云服务模式冲突 |

### 阶段 D：离线 contextual bandit（RL 的第一步，~2-3 天）

| # | 任务 |
|---|---|
| D.1 | schema 加 `candidates` 反事实表（§2.6a） |
| D.2 | 实现 `MockRouterEnv` + 可插拔 `compute_reward` |
| D.3 | **LinUCB / 离线 Thompson sampling**，不是 PPO/GRPO |
| D.4 | 用 IPS / doubly-robust 做 off-policy 评估，与 SFT 基线对比 |

**只有 D.4 显示 bandit 真的超过 SFT，才考虑 policy gradient。** 在此之前不要引入 critic / reference model / rollout engine。

### 明确不做的事

- ❌ 现在实现 REINFORCE / PPO / GRPO
- ❌ 现在做模型量化、跨平台导出、iOS/Android SDK（模型 84KB，没有可优化的负载）
- ✅ 已补可选 tiny Transformer 对照分支（仅在 pooled 已证明的边界难例上比较；默认仍是 pooled）
- ✅ 已补本地文学文本切块入口；《局外人》只从用户本地文件读取，不进仓库
- ❌ 照抄 vision 文档里的 8MB / 40MB / 三段式置信度权重（那份文档的技术描述全是过期的，见 `CODE_MAP.md` §6）

---

## 5. 与导师指示的对照

| 导师指示 | 采纳情况 | 说明 |
|---|---|---|
| RL 是 Jev 的一个阶段 | ✅ 方向采纳，⏸ 时序后置 | 核实后：只有论文 A（2503.23303）是 RL，而 MiniJev 实现的是论文 B（纯监督）。RL 缺的是 reward 不是算法（§2.3） |
| 分别实现 training / inference pipeline | ✅ 采纳，⚠️ 形式修正 | 7 个开源实现**没有一个**用双目录树，应该是两个入口 + 共享核心（§3） |
| 用 tinygrad 实现 | ✅ 已经在做 | 保持。注意：tinygrad 生态里**没人**做过 RL 路由，这确实是差异点 |
| 后期做硬件性能优化 | ⏸ 后置 | 诚实的说法：20,933 参数 / 84KB 目前没有可优化的负载。等模型大到延迟有意义时再谈 |
| 项目要 make difference | ✅ 强烈同意 | 目前的「difference」就是 **21.45% vs 98%** 这个落差本身。先把它变成可复现的诚实数字 |
| 先做最基础版本，慢慢改，需要 dirty work | ✅ 强烈同意 | 阶段 A 已按此收紧为「最小可信闭环」，不做完整测试框架 |
| 先调研其他开源实现的模块 | ✅ 已完成 | 7 个仓库，结论见 §3 |
| 预留 RL 引入 | ✅ 采纳 | 具体到 schema 字段、policy 输出、reward 接口三层（§2.6） |

---

## 附：本轮新增的文件

| 文件 | 用途 | 处置建议 |
|---|---|---|
| `data/external/massive/convert_massive.py` | MASSIVE → MiniJev JSONL 转换器，复用 `evaluate_massive.py` 的 mapping（保证单变量对照） | 阶段 B 正式化为 `minijev/convert.py` |
| `data/external/massive/derived/massive_{train,val,test}.jsonl` | 转换产物（11,514 / 2,033 / 2,974） | 加进 `.gitignore` |
| `models/massive_sft.safetensors` + `massive_sft_metrics.json` | 微调结果，79.15% 的证据 | 保留 |
| `models/massive_zh_test_metrics.json` | 零样本 21.45% 的证据 | 保留 |

**质量门**：`ruff check .` ✅ / `ruff format --check .` ✅（20 files）。
