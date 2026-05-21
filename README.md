# Constraint-Aware Hierarchical Search

An LLM-powered hierarchical regulation tree search and QA system.

---

## Quick Start

### 1. Install

```bash
conda create -n constraint-search python=3.11 -y
conda activate constraint-search
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your API credentials:

```env
OPENAI_CHAT_BASE_URL=https://api.openai.com/v1
OPENAI_CHAT_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Prepare Data

Place your regulation text files in `input/` (`.txt` format).

### 4. Generate Rules

Different regulation documents vary in hierarchical structure, coding format, and chunking granularity. This step asks the LLM to analyze your text samples, automatically identify numbering rules, hierarchy depth, exclusion patterns, etc., and generate extraction rules accordingly — resulting in better tree quality.

```bash
python scripts/analyze_and_write_rules.py --dataset-name my_dataset --input-path input/
```

Output goes to `rules/my_dataset/`.

### 5. Build Tree Index

```bash
python scripts/regtree_cli.py build --dataset-name my_dataset
```

Output goes to `rag-storage/my_dataset/`. Add `--resume` to continue an interrupted build from checkpoint.

### 6. Query

```bash
python scripts/regtree_cli.py query --dataset-name my_dataset "titanium dioxide; TiO2 content >= 80%"
```

---

## Algorithm

### LLM Candidate Selection (F<sub>θ</sub>)

At each layer of the hierarchical tree search, the system invokes an LLM selection function:

**(v<sub>t+1</sub>, σ<sub>t</sub>, ρ<sub>t</sub>, η<sub>t</sub>) = F<sub>θ</sub>(x, v<sub>t</sub>, {Γ(u)}<sub>u ∈ C<sub>t</sub>(x)</sub>)**

| Symbol | Meaning | Prompt / Code |
|--------|---------|---------------|
| x | User query | `query` field |
| v<sub>t</sub> | Current node | `current_node` field |
| {Γ(u)} | Candidate packages | `candidates` field |
| **F<sub>θ</sub>** | **LLM selection function** | **[System prompt → `prompts.py:20`](regtree_agent/prompts.py#L20) · [Task prompt → `prompts.py:37`](regtree_agent/prompts.py#L37) · [Implementation → `search.py:1188`](regtree_agent/search.py#L1188)** |
| v<sub>t+1</sub> | Selected node (`selected_id`) | [`output_schema.selected_id`](regtree_agent/search.py#L1222) |
| σ<sub>t</sub> | Stop signal | [`output_schema.stop`](regtree_agent/search.py#L1223) |
| ρ<sub>t</sub> | Confidence | [`output_schema.confidence`](regtree_agent/search.py#L1224) |
| η<sub>t</sub> | Rationale | [`output_schema.reason`](regtree_agent/search.py#L1225) |

---

## Project Structure

```
input/              Source regulation text files (.txt)
rules/              Auto-generated extraction rules
rag-storage/        Build artifacts (tree + vectors)
scripts/            CLI entry points
regtree_agent/      Core library code
```

## Notes

- Python >= 3.11
- Both building and querying consume LLM API tokens
- Use `--dataset-name` to isolate different datasets
