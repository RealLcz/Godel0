# Gödel0

Self-improving coding agent with built-in Proposer and SWE-smith, implementing an approximation of the Gödel Machine.

## Overview

Gödel0 evolves a coding agent through a tree search where each node is a complete Agent code version consisting of:

- **Solver**: Solves coding tasks (inherited from DGM/HGM baseline)
- **Proposer**: Generates new coding tasks by introducing bugs
- **SWE-smith Engine**: Creates bug candidates via procedural and LM-based mutations
- **Shared Tools**: Common toolset used by both Solver and Proposer

The framework separates a **trusted control layer** (tree search, isolation, evaluation, scoring) from an **evolvable Agent code layer** (Solver, Proposer, tools, prompts).

## Architecture

```
godel0/
├── src/godel0/          # Trusted control layer (not evolvable)
│   ├── controller/      # Orchestrator, budget, scorer, resume
│   ├── tree/            # Node tree, archive, selection
│   ├── evolution/       # Cycle builder, detectors, diagnosis, self-edit
│   ├── evaluation/      # Level 1/2 evaluators, solver runner
│   ├── proposer_trusted/# Candidate validator, safety, auditor
│   ├── tasks/           # TaskStore, batch builder, repo pool
│   ├── execution/       # Subprocess + Apptainer backends
│   ├── git/             # Worktree, refs, patch utils
│   ├── storage/         # Atomic JSON/JSONL, paths, events
│   └── schemas/         # Pydantic models for all data structures
├── initial_agent/src/   # Evolvable Agent code (Root Node)
│   ├── coding_agent.py  # Solver core (from HGM)
│   ├── llm.py           # Multi-model LLM client (Qwen/Minimax/DeepSeek/OpenAI)
│   ├── llm_withtools.py # Tool-calling agent loop
│   ├── tools/           # Shared tools (bash, editor)
│   ├── proposer/        # Task generation policy
│   └── swesmith/        # Bug candidate engine
├── configs/             # YAML configurations
├── tests/               # Unit, integration, and E2E tests
└── scripts/             # Export, verify, validate scripts
```

## Supported Models

Gödel0 supports multiple open-source and commercial models:

| Model Family | Prefix | Environment Variables |
|---|---|---|
| DeepSeek | `deepseek/*` | `DEEPSEEK_API_KEY`, `DEEPSEEK_API_BASE_URL` |
| Qwen | `qwen/*` | `QWEN_API_KEY`, `QWEN_API_BASE_URL` or `VLLM_HOST`/`VLLM_PORT` |
| Minimax | `minimax/*` | `MINIMAX_API_KEY`, `MINIMAX_API_BASE_URL` |
| OpenAI | `gpt-*`, `o*` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/*` | `OpenRouter_API_KEY` |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Export Solver Core from HGM
python scripts/export_solver_core.py --source-repo /path/to/HGM --output initial_agent/src

# Verify Solver Core parity
python scripts/verify_solver_core.py

# Run tests
PYTHONPATH=src:. python -m pytest tests/ -v

# Run evolution
python -m godel0.cli run --config configs/local_smoke.yaml
```

## Scoring

Node score = a × b, where:
- `a = λr + (1-λ)p` (solver score: retention × frontier accuracy)
- `b = max(0, 1 - 2|p - 0.5|)` (proposer score: difficulty calibration)
- `r` = Level 1 retention rate
- `p` = Level 2 frontier accuracy
- `λ` = regression weight (default 0.5)

## Key Design Principles

1. **One Agent Workflow**: Solver, Proposer, and Self-evolution all use the same `coding_agent.py`
2. **Trusted Boundary**: Agent proposes, trusted controller verifies
3. **Joint Diagnosis**: No pre-selection of Solver vs Proposer; diagnose the full cycle
4. **Git-based Versioning**: Each node is a git commit with a ref
5. **F2P Validation**: Candidates must produce Fail-to-Pass tests in a clean workspace

## Testing

```bash
# Unit tests (110 tests)
PYTHONPATH=src:. python -m pytest tests/unit/ -v

# Integration tests (9 tests)
PYTHONPATH=src:. python -m pytest tests/integration/ -v

# E2E test (1 test - full toy repo cycle)
PYTHONPATH=src:. python -m pytest tests/e2e/ -v -s

# All tests
PYTHONPATH=src:. python -m pytest tests/ -v
```

## License

Apache-2.0
