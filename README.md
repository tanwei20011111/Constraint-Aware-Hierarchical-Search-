# RegTree Agent

基于大模型的法规树检索与问答系统。支持从法规文本自动构建层级树结构，并通过多轮检索对话回答 HS 编码等归类问题。

---

## 快速开始

### 1. 环境准备

```bash
# 安装依赖（假设使用 poetry 或 pip）
pip install -r requirements.txt

# 配置环境变量：复制 .env.example 为 .env 并填入你的 API 密钥
cp .env.example .env
```

`.env` 需要配置：

```env
OPENAI_CHAT_BASE_URL=https://api.openai.com/v1
OPENAI_CHAT_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

OPENAI_EMBEDDING_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

OPENAI_TEMPERATURE=0
OPENAI_MAX_TOKENS=4000
```

### 2. 准备输入文件

将要分析的法规文本放入 `input/` 目录，每个文件一个 `.txt`：

```
input/
├── 第一章.txt
├── 第二章.txt
└── ...
```

### 3. 生成规则（一次性）

让大模型自动分析文本结构，生成建树所需的规则文件：

```bash
python scripts/analyze_and_write_rules.py \
  --dataset-name my_dataset \
  --input-path input/
```

输出到 `rules/my_dataset/` 目录，包含：
- `rule_profiles.json` — 规则配置
- `rule_map.json` — 文件映射
- `rule_analysis.json` — 结构分析报告

### 4. 构建树索引

使用 LLM 从文本构建层级树并生成向量：

```bash
python scripts/regtree_cli.py build \
  --dataset-name my_dataset \
  --resume
```

产物保存在 `rag-storage/my_dataset/`：
- `regtree_tree.json` — 树结构
- `regtree_vectors.npz` — 向量文件

支持断点续建（`--resume`），中断后重新运行会自动恢复。

### 5. 查询问答

```bash
python scripts/regtree_cli.py query \
  --dataset-name my_dataset \
  "颜料（金红石型钛白粉）；二氧化钛≥80%；生产油漆用"
```

---

## 核心概念

### 文件结构

```
.
├── input/                  # 原始法规文本（.txt）
├── rules/                  # 自动生成的规则
│   └── <dataset_name>/
│       ├── rule_profiles.json
│       ├── rule_map.json
│       └── rule_analysis.json
├── rag-storage/            # 构建产物
│   └── <dataset_name>/
│       ├── regtree_tree.json
│       └── regtree_vectors.npz
├── scripts/
│   ├── regtree_cli.py      # 统一 CLI（build + query）
│   ├── analyze_and_write_rules.py  # 规则生成
│   └── evaluate_hscode_xlsx.py     # 批量评测
└── regtree_agent/
    ├── search.py           # 检索引擎
    ├── tree_index.py       # 建树引擎
    ├── online.py           # API 客户端
    └── config.py           # 配置管理
```

### 建树流程

1. **规则分析**：LLM 分析样本文本 → 生成 `hierarchy_hints` + `code_format_hint`
2. **文本切窗**：按语义边界（段落→句子）切分，生成 evidence chunks
3. **结构抽取**：LLM 逐块抽取层级节点（code, title, children, exclusions, definitions）
4. **证据挂接**：语义匹配节点与 evidence chunks
5. **节点去重**：合并 code+title 重复的节点
6. **向量生成**：节点、chunks、segments 统一 embedding

### 检索流程

1. **锚点提取**：LLM 从 query 提取 anchor_terms（商品本体）和 constraint_terms（限定条件）
2. **全局召回**：depth=0 时从整棵树语义检索 top-K 节点（加权融合：node 0.5 + segment 0.25 + keyword 0.25）
3. **树结构下钻**：depth≥1 时沿 RegTree 子节点展开，不再跨分支召回
4. **LLM 选择**：每层候选由 LLM 选择最佳下一跳
5. **多轮搜索**：高置信停止，否则 planner 决定 refine/switch/compare
6. **答案生成**：基于检索到的节点和 evidence 生成最终编码

---

## 详细用法

### 规则生成

```bash
# 基本用法
python scripts/analyze_and_write_rules.py \
  --dataset-name hs_taxnote \
  --input-path input/

# 指定参考规则（可选）
python scripts/analyze_and_write_rules.py \
  --dataset-name hs_taxnote \
  --input-path input/ \
  --base-rule-file rules/default/rule_profiles.json \
  --base-profile hs_code

# 只打印结果不写文件（dry run）
python scripts/analyze_and_write_rules.py \
  --dataset-name hs_taxnote \
  --input-path input/ \
  --dry-run

# 调整样本数量（默认12个 chunk）
python scripts/analyze_and_write_rules.py \
  --dataset-name hs_taxnote \
  --input-path input/ \
  --sample-count 20
```

### 建树

```bash
# 全量构建
python scripts/regtree_cli.py build --dataset-name hs_taxnote

# 断点续建（推荐）
python scripts/regtree_cli.py build --dataset-name hs_taxnote --resume

# 从 checkpoint 重新开始
python scripts/regtree_cli.py build --dataset-name hs_taxnote --no-resume

# 只重建向量（复用已有 tree JSON）
python scripts/regtree_cli.py build --dataset-name hs_taxnote --reuse-tree

# 调整 segment 切分粒度
python scripts/regtree_cli.py build \
  --dataset-name hs_taxnote \
  --segment-chars 200 \
  --segment-overlap 10

# 打印 LLM 结构抽取过程（调试）
python scripts/regtree_cli.py build \
  --dataset-name hs_taxnote \
  --print-llm-units \
  --print-llm-window-split
```

### 查询

```bash
# 基本查询
python scripts/regtree_cli.py query \
  --dataset-name hs_taxnote \
  "蒸汽锅炉：蒸发量超过45吨/时的水管锅炉"

# 调整搜索参数
python scripts/regtree_cli.py query \
  --dataset-name hs_taxnote \
  --branch-top-k 5 \
  --global-top-k 5 \
  --max-depth 8 \
  --max-rounds 5 \
  "涡轮喷气发动机：推力超过25千牛顿"

# 打印调试信息
python scripts/regtree_cli.py query \
  --dataset-name hs_taxnote \
  --print-llm-inputs \
  --print-similarity-trace \
  --print-answer-trace \
  "阳离子改性瓜尔胶；由瓜尔豆通过化学改性处理获得；黏度300-1000 mPa.s；工业级"

# 静默模式（不输出进度）
python scripts/regtree_cli.py query \
  --dataset-name hs_taxnote \
  --quiet \
  "蒸汽锅炉"
```

### 批量评测

```bash
python scripts/evaluate_hscode_xlsx.py \
  --input data/test/品目注释测试.xlsx \
  --output data/test/预测结果.xlsx \
  --dataset-name hs_taxnote
```

---

## 高级配置

### 搜索参数调优

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `branch_top_k` | 5 | 每层保留的子节点候选数 |
| `global_top_k` | 5 | depth=0 时全局召回的 top-K 节点数 |
| `max_depth` | 8 | 单轮最大下钻深度 |
| `max_rounds` | 5 | 多轮搜索最大轮数 |
| `candidate_field_mode` | full | 候选节点字段丰富度 |

### Segment 切分参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `segment-chars` | 200 | 每个 segment 最大字符数 |
| `segment-overlap` | 10 | 相邻 segment 重叠字符数 |

语义边界优先：段落 → 句子 → 固定长度滑动窗口。

---

## 注意事项

1. **LLM 调用成本**：建树和查询都会调用大模型 API，注意控制 token 消耗
2. **Checkpoint**：建树过程会自动保存 checkpoint，中断后加 `--resume` 可恢复
3. **数据集隔离**：不同数据集使用 `--dataset-name` 隔离，产物互不干扰
4. **规则复用**：同一套规则可用于多个相似数据集的建树

---

## 许可证

MIT
