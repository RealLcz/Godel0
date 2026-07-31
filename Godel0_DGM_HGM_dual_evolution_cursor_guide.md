# Godel0：对齐 DGM / HGM 的 Dual Self-Evolution 改造指南

> 目标读者：Cursor / Codex 等代码代理  
> 目标仓库：`RealLcz/Godel0`  
> 核心目标：保留 Godel0 当前的 **Proposer → Solver 双阶段进化**，但让每个阶段尽量遵循 DGM/HGM 的 self-improvement 范式：
>
> **一个真实失败 entry → 一个聚焦 diagnosis → 一次通用代码修改 → 仅做机械合法性检查 → 用真实评估决定修改优劣。**

---

## 1. 总体目标

当前 Godel0 已经具备：

```text
选择 parent
→ 选择 proposer failure
→ proposer diagnosis
→ proposer self-edit
→ 选择 solver failure
→ solver diagnosis
→ solver self-edit
→ 创建 child
→ Level 1 / Proposer / Level 2 评估
```

整体方向正确，但当前实现存在四类问题：

1. **基础设施问题导致所有 child 被机械拒绝**
   - tracked `__pycache__/*.pyc` 被写入累计 diff；
   - proposer phase 的非法修改要等 solver phase 完成后才被发现；
   - `allow_empty_phase=True` 允许某个阶段没有真正进化。

2. **Diagnosis prompt 过度诱导大规模重构**
   - 多次强调 “implement the full diagnosis”；
   - 多次强调 “address this class of failures”；
   - 鼓励多文件、完整机制，导致一次 mutation 经常修改 500–1200 行。

3. **Proposer diagnosis 容易把 trusted validator 当成可绕过的障碍**
   - 在 validation exception 后返回 `passed=True`；
   - 跳过 causal ablation；
   - 降低阈值；
   - 伪造或修饰 validation metadata。

4. **失败 mutation 没有形成可追踪反馈**
   - 同一个 proposer candidate 被连续多次选择；
   - PatchGuard failure 没有进入下一轮 diagnosis；
   - diagnosis 解析失败后仍使用宽泛 fallback，触发无依据的大规模修改。

本次改造应恢复 DGM/HGM 的核心：

```text
一个失败样本只是诊断证据；
改动必须具有一般性，但不要求建设完整子系统；
机械合法的 child 应进入真实评估；
能力好坏由 rollout 和树搜索决定，而不是由 prompt 或静态启发式决定。
```

---

# 2. 目标 Dual-HGM 流程

最终流程应调整为：

```text
Parent node
│
├── Proposer phase
│   ├── 选择一个未优先尝试过的 proposer failure entry
│   ├── 只诊断一个 primary failure mechanism
│   ├── 生成一个聚焦、通用的 improvement issue
│   ├── coding agent 修改 proposer
│   ├── 清理 runtime artifacts
│   ├── proposer-specific mechanical gates
│   └── intermediate commit
│
├── Solver phase
│   ├── 选择一个未优先尝试过的 solver failure entry
│   ├── 基于 proposer intermediate commit 读取当前代码
│   ├── 只诊断一个 primary failure mechanism
│   ├── coding agent 修改 solver
│   ├── 清理 runtime artifacts
│   ├── solver-specific mechanical gates
│   └── final commit
│
└── Child evaluation
    ├── Level 1 retention
    ├── child proposer 生成 K 个任务
    ├── child solver 求解 K 个任务
    ├── 计算 proposer / solver utility
    └── 进入 archive / tree selection
```

必须避免：

```text
先在 parent 上同时生成 proposer diagnosis 和 solver diagnosis
→ proposer 修改代码
→ solver 仍按旧代码 diagnosis 执行
```

更合理的顺序是：

```text
proposer diagnose
→ proposer edit
→ intermediate commit
→ solver diagnose 当前 intermediate code
→ solver edit
```

---

# 3. 修改优先级

## P0：先修复 child 无法创建的问题

必须首先完成：

1. 清理 tracked runtime artifacts；
2. 设置 `PYTHONDONTWRITEBYTECODE=1`；
3. proposer 和 solver phase 分别做 PatchGuard；
4. `allow_empty_phase=False`；
5. `proposer/schemas.py` 解冻；
6. `proposer/request.py` 保持冻结。

在 P0 完成前，不要继续大规模 evolution 实验。

## P1：改造 Prompt

完成：

1. Diagnosis 强制一个 primary root cause；
2. 删除鼓励“大而全修改”的表述；
3. Proposer prompt 明确 trusted validator 不可绕过；
4. Solver prompt 明确不要为单个 task 建设大系统；
5. 删除 generic fallback diagnosis；
6. 缩小并条件化 code dump。

## P2：恢复可持续搜索

完成：

1. 同一 parent 下优先选择未尝试 entry；
2. 保存 failed mutation attempt；
3. 将 mutation failure 反馈给后续选择和诊断；
4. 机械合法 child 进入真实评估；
5. 让评估而非静态启发式决定 child 价值。

---

# 4. P0-1：清理 runtime artifacts

## 修改文件

```text
src/godel0/git/repository.py
src/godel0/evolution/self_edit.py
src/godel0/evolution/child_builder.py
src/godel0/controller/orchestrator.py
initial_agent/src/.gitignore
```

## 4.1 增加统一 runtime artifact 判断

在 `src/godel0/git/repository.py` 中增加：

```python
TRANSIENT_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

TRANSIENT_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".backup",
    ".bak",
)


def is_runtime_artifact(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = normalized.split("/")
    return (
        any(part in TRANSIENT_DIR_NAMES for part in parts)
        or normalized.endswith(TRANSIENT_SUFFIXES)
    )
```

将现有 `_is_transient_untracked_path()` 合并到该函数，避免出现两套规则。

## 4.2 增加 restore 函数

```python
def restore_runtime_artifacts(repo_path: Path, base_commit: str) -> None:
    """Remove runtime-only changes before constructing an evolution patch."""

    tracked = run_git(repo_path, "ls-files", "-z").stdout.split("\0")
    tracked_runtime = [
        path
        for path in tracked
        if path and is_runtime_artifact(path)
    ]

    if tracked_runtime:
        run_git(
            repo_path,
            "restore",
            "--source",
            base_commit,
            "--worktree",
            "--",
            *tracked_runtime,
        )

    untracked = run_git(
        repo_path,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).stdout.split("\0")

    for relative_path in untracked:
        if not relative_path or not is_runtime_artifact(relative_path):
            continue

        path = repo_path / relative_path
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            import shutil
            shutil.rmtree(path, ignore_errors=True)
```

## 4.3 调用位置

必须在以下位置调用：

```text
SelfEditRunner._patch_problem()
proposer phase diff 计算前
proposer intermediate commit 前
solver phase diff 计算前
final cumulative diff 计算前
child gate 运行后、最终 commit 前
```

示例：

```python
restore_runtime_artifacts(worktree, base_commit)
patch = diff_vs_commit(worktree, base_commit)
```

## 4.4 禁止 Python 写 bytecode

所有执行 self-edit、import gate 和 tests 的 subprocess 环境中设置：

```python
env["PYTHONDONTWRITEBYTECODE"] = "1"
```

可以同时使用：

```bash
python -B ...
```

但环境变量应作为统一保证。

## 4.5 `.gitignore`

在 `initial_agent/src/.gitignore` 中加入：

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.backup
*.bak
chat_history.md
model_patch.diff
```

## 4.6 Root 初始化前检查

在 `_initialize_root()` 中，在创建 root commit 前清理 transient artifacts。

新增：

```python
def remove_runtime_artifacts_from_tree(repo_root: Path) -> None:
    ...
```

Root gate 还应检查：

```bash
git ls-files
```

不得包含：

```text
__pycache__
.pyc
.pyo
.pytest_cache
.backup
.bak
```

---

# 5. P0-2：Proposer 和 Solver 分阶段 Gate

## 修改文件

```text
src/godel0/evolution/child_builder.py
src/godel0/evolution/patch_guard.py
src/godel0/constants.py
src/godel0/config.py
```

## 5.1 Role-specific allowlist

在 `constants.py` 中定义：

```python
PROPOSER_ALLOWED_PATCH_PREFIXES = (
    "proposer/",
    "swesmith/",
)

SOLVER_ALLOWED_PATCH_PREFIXES = (
    "coding_agent.py",
    "llm.py",
    "llm_withtools.py",
    "tools/",
    "prompts/",
    "utils/",
    "requirements.txt",
)
```

当前阶段不建议 proposer 修改：

```text
utils/
llm.py
llm_withtools.py
coding_agent.py
tools/
```

原因是：

- dual evolution 已经有独立 solver phase；
- proposer 修改 shared runtime 后，会增加 solver diagnosis 的不确定性；
- 更难区分 proposer improvement 与 solver improvement 的贡献。

未来可以设计 shared phase，但本次不要混入。

## 5.2 Frozen files

修改：

```python
FORBIDDEN_PATCH_PATTERNS = (
    "../",
    "/.git",
    "symlink",
    "proposer/request.py",
)
```

删除：

```python
"proposer/schemas.py"
```

### 规则

- `proposer/request.py`：永久冻结；
- `proposer/schemas.py`：允许进化，但必须做兼容性检查；
- trusted control layer：不在 agent worktree 中，天然不可修改；
- agent `tests/`：建议从 evolution allowlist 中去除。

## 5.3 Phase gate

在 `ChildBuilder.build_dual()` 中改为：

```text
proposer self-edit
→ cleanup runtime artifacts
→ proposer PatchGuard
→ syntax gate
→ proposer import/schema gate
→ intermediate commit

solver self-edit
→ cleanup runtime artifacts
→ solver PatchGuard
→ syntax gate
→ solver import gate
→ final cumulative patch
```

伪代码：

```python
if proposer_diagnosis is not None:
    proposer_result = run_self_edit(...)

    restore_runtime_artifacts(worktree, phase_base)
    proposer_patch = diff_vs_commit(worktree, phase_base)

    phase_errors = self._validate_phase_patch(
        role="proposer",
        worktree=worktree,
        patch=proposer_patch,
        base_commit=phase_base,
    )
    if phase_errors:
        return failed_result(...)

    phase_base = git_commit(...)

if solver_diagnosis is not None:
    solver_result = run_self_edit(...)

    restore_runtime_artifacts(worktree, phase_base)
    solver_patch = diff_vs_commit(worktree, phase_base)

    phase_errors = self._validate_phase_patch(
        role="solver",
        ...
    )
    if phase_errors:
        return failed_result(...)
```

## 5.4 禁止 empty phase

将：

```python
allow_empty_phase=True
```

改为：

```python
allow_empty_phase=False
```

语义：

```text
proposer phase 必须产生 source patch；
solver phase 必须产生 source patch；
任一阶段失败，本次 dual mutation 失败。
```

不要让 proposer 失败后继续花费一次 solver self-edit。

---

# 6. P0-3：`proposer/schemas.py` 兼容性 Gate

## 修改文件

```text
src/godel0/evolution/gates.py
tests/unit/evolution/test_gates.py
```

当前 `ProposerExtensionGate` 中：

```python
report.schemas_parseable = True
```

是无条件通过，必须改掉。

## 6.1 必须保留的类

```python
REQUIRED_SCHEMA_CLASSES = {
    "BugConstraints",
    "FailureSignature",
    "BugGenerationPlan",
    "CodeTarget",
}
```

## 6.2 兼容性规则

允许：

- 新增字段；
- 新增 class；
- 新增 optional planning metadata；
- 新字段有默认值；
- 新增内部 strategy 表示。

禁止：

- 删除原有 class；
- 删除原有关键字段；
- 将原有 optional/default 字段改为无默认 required；
- 改变旧 JSON payload 的可解析性；
- 增加 `skip_validation`、`force_accept`、`accept_on_error` 一类绕过 trusted validation 的语义。

## 6.3 Gate 示例

```python
def validate_proposer_schema_compatibility(node_code_dir: Path) -> list[str]:
    errors: list[str] = []

    schemas_path = node_code_dir / "proposer" / "schemas.py"
    if not schemas_path.is_file():
        return ["proposer/schemas.py is missing"]

    module = load_module_from_path(
        "candidate_proposer_schemas",
        schemas_path,
    )

    for class_name in REQUIRED_SCHEMA_CLASSES:
        if not hasattr(module, class_name):
            errors.append(f"missing schema class: {class_name}")

    if errors:
        return errors

    legacy_plan_payload = {
        "plan_id": "compat-plan",
        "target_repo_id": "repo",
        "target_base_commit": "abc",
        "target_file": "src/example.py",
        "strategy": "repo_chain",
    }

    try:
        module.BugGenerationPlan.model_validate(legacy_plan_payload)
    except Exception as exc:
        errors.append(
            "BugGenerationPlan no longer accepts a legacy payload: "
            f"{exc}"
        )

    legacy_constraints_payload = {
        "min_modified_files": 1,
        "max_modified_files": 1,
        "max_modified_lines": 20,
    }

    try:
        module.BugConstraints.model_validate(legacy_constraints_payload)
    except Exception as exc:
        errors.append(
            "BugConstraints no longer accepts a legacy payload: "
            f"{exc}"
        )

    return errors
```

## 6.4 不要冻结内部表示

不要因为 schema 重要，就永久冻结 `proposer/schemas.py`。

Proposer 的能力进化可能合理依赖：

- 更丰富的 failure signature；
- 更明确的 multi-file causal roles；
- 更好的 mutation constraints；
- 更强的 planning representation。

应冻结的是外部通信协议，而不是内部认知表示。

---

# 7. P1-1：重写 Diagnosis System Prompt

## 修改文件

```text
src/godel0/evolution/hgm_diagnose.py
```

替换当前 `DIAGNOSE_SYSTEM_TEMPLATE`。

## 推荐版本

```python
DIAGNOSE_SYSTEM_TEMPLATE = """You are diagnosing one observed failure of an
evolvable software agent.

Your goal is to identify exactly ONE general capability defect in the current
agent implementation and propose exactly ONE coherent improvement.

The observed failure is evidence for the diagnosis, not the target of the
modification. Do not propose a task-specific fix, repository-specific rule,
hard-coded identifier, expected patch, or benchmark-specific behavior.

Prefer the smallest coherent mechanism that could plausibly have changed the
observed trajectory. A general improvement may touch multiple production files
when necessary, but breadth is not a virtue and a new subsystem is not required.

Distinguish carefully between:

1. a capability failure in the evolvable agent;
2. an infrastructure or execution failure;
3. a rejection produced by a trusted evaluator.

Do not convert infrastructure failures into agent capability proposals.
Do not weaken, bypass, redefine, or imitate trusted evaluation behavior.

You will be shown a focused subset of the current agent implementation and one
failure entry. Ground the root cause and proposed edit in the supplied evidence.

{role_constraints}

# Relevant Agent Implementation
----- Agent Implementation Start -----
{code}
----- Agent Implementation End -----
"""
```

---

# 8. P1-2：Proposer Diagnosis Prompt

## Role constraint

```python
PROPOSER_DIAGNOSIS_CONSTRAINTS = """You are diagnosing the Proposer.

The trusted candidate validator defines the task specification. It is not an
obstacle for the Proposer to bypass.

Never propose or implement any of the following:

- marking a candidate as passed after an exception;
- accepting a candidate when causal isolation could not be executed;
- skipping causal isolation or causal ablation;
- lowering trusted acceptance thresholds;
- fabricating validation metadata or causal statistics;
- changing trusted pass/fail semantics;
- hiding answer leakage through superficial lexical rewriting alone;
- editing proposer/request.py;
- editing trusted-controller code.

A rejected candidate is negative evidence about upstream generation. Improve
target selection, planning, patch construction, mutation execution, issue
generation, or candidate robustness so that future candidates genuinely satisfy
the existing validation contract.

Choose one primary failure mechanism. If the candidate has multiple rejection
reasons, select the earliest causal reason that explains the later failures.
"""
```

## Primary rejection priority

在代码中将多个 rejection reason 归并为一个 primary reason，推荐优先级：

```text
1. execution/setup failure
2. malformed or non-applicable patch
3. syntax/import failure
4. no Fail-to-Pass behavior
5. causal isolation failure
6. statement quality / answer leakage
7. duplicate / calibration / diversity failure
```

原因：

- 如果 patch 无法 apply，后续 causal failure 和 statement failure 都不是主要原因；
- 一次 diagnosis 只解决一个最早 causal failure；
- 避免一次 evolution 同时重写 planner、patch engine、statement generator 和 validator adapter。

---

# 9. P1-3：Solver Diagnosis Prompt

```python
SOLVER_DIAGNOSIS_CONSTRAINTS = """You are diagnosing the Solver.

Improve the Solver's general coding behavior. Relevant edit surfaces may include:

- the live workflow in coding_agent.py;
- prompts used by the live forward path;
- existing tool descriptions or implementations;
- context management and tool-loop behavior;
- verification or testing behavior already present in the agent.

Do not:

- encode the observed task, repository, file, symbol, or expected patch;
- copy private-test behavior into the agent;
- force one particular tool on every task unless the evidence demonstrates a
  general workflow failure;
- build a large evaluation framework when a focused prompt, workflow, or
  existing-tool change addresses the observed defect;
- modify benchmark, trusted evaluation, or task-generation code.

Choose exactly one capability defect supported by the trajectory. Prefer a
focused change to the current live workflow over introducing a new subsystem.
"""
```

---

# 10. P1-4：Diagnosis User Prompt 和输出 Schema

当前 prompt 中的：

```text
potential_improvements
```

会鼓励模型先大范围发散。

改为更收敛的结构：

```python
DIAGNOSE_USER_TEMPLATE = """{intro}

# Failure Case
----- Failure Case Start -----
{github_issue}
----- Failure Case End -----

# Agent Run Log
----- Agent Run Log Start -----
{md_log}
----- Agent Run Log End -----

# Generated / Predicted Patch
----- Patch Start -----
{predicted_patch}
----- Patch End -----

# Evaluation / Validation Result
----- Evaluation Start -----
{eval_log}
----- Evaluation End -----

Return one JSON object between the required JSON markers.

The JSON must contain:

- "failure_summary":
  Briefly state what happened in this run.

- "primary_root_cause":
  Exactly one agent capability defect supported by the evidence.

- "generalization":
  Explain why the defect is broader than this one instance.

- "single_improvement":
  Exactly one coherent capability improvement.

- "edit_scope":
  A list of the live files or components most likely to require changes.
  Keep this focused.

- "implementation_suggestion":
  A concrete implementation direction. Prefer adapting an existing workflow,
  prompt, or tool over creating a new subsystem.

- "expected_behavior_change":
  State what should be observably different in a future run.

- "problem_description":
  A concise GitHub-issue-style task that another coding agent can implement.

Do not include multiple alternative improvements. Do not include task-specific
identifiers as implementation constants.
"""
```

## Parser validation

`parse_diagnose_json()` 应验证：

```python
required = {
    "primary_root_cause",
    "single_improvement",
    "implementation_suggestion",
    "expected_behavior_change",
    "problem_description",
}
```

并验证：

```python
edit_scope = data.get("edit_scope") or []
if len(edit_scope) > 4:
    raise SchemaValidationError(
        "diagnosis edit_scope must contain at most four focused components"
    )
```

这里的 `4` 是 diagnosis 聚焦约束，不是最终 patch 的绝对文件上限。

---

# 11. P1-5：拒绝危险 Diagnosis

新增 semantic guard：

```python
DANGEROUS_DIAGNOSIS_PATTERNS = (
    "passed=true",
    "passed = true",
    "accept on exception",
    "accept after exception",
    "skip causal",
    "bypass validation",
    "lower the threshold",
    "reduce the threshold",
    "force accept",
    "fabricate",
    "proposer/request.py",
)
```

检查时使用 lowercase。

注意：

- 不要简单拒绝所有包含 `threshold` 的 diagnosis；
- 只拒绝“降低 trusted acceptance threshold”语义；
- 可以先做简单 substring guard，再在 prompt 中强化；
- 后续可增加一个 deterministic structured field：

```json
{
  "trusted_boundary_effect": "preserve"
}
```

仅允许：

```text
preserve
```

---

# 12. P1-6：删除 Generic Fallback Diagnosis

当前 `HgmEntryDiagnoser._fallback()` 会在 diagnosis 失败后生成宽泛任务，例如：

```text
inspect proposer/ and swesmith/
make the mechanism robust
```

必须删除这种行为。

改为：

```python
def diagnose_proposer(...) -> Optional[CycleDiagnosis]:
    ...
    if all retries fail:
        return None
```

同理：

```python
def diagnose_solver(...) -> Optional[CycleDiagnosis]:
    ...
```

Orchestrator 处理：

```python
if proposer_entry is not None and proposer_diagnosis is None:
    record_mutation_failure(
        stage="proposer_diagnosis",
        ...
    )
    return failed_child_result
```

DGM/HGM 风格更接近：

```text
diagnosis 失败
→ 本次 mutation 失败
```

而不是：

```text
diagnosis 失败
→ 用没有证据的宽泛指令继续改代码
```

---

# 13. P1-7：重写 Self-Edit Prompt

## 修改文件

```text
src/godel0/evolution/self_edit.py
```

替换当前 `EDIT_PROTOCOL`。

## 推荐版本

```python
EDIT_PROTOCOL = """
Implement the improvement task in the current agent repository.

Inspect the existing implementation before editing. Make the simplest coherent
change that implements the requested capability improvement in the live runtime
path.

Do not hard-code identifiers or behavior from the observed failure instance.
Do not modify trusted evaluation code, frozen transport protocols, generated
artifacts, documentation, backup files, or unrelated tests.

Prefer modifying an existing workflow, prompt, or tool over introducing a new
subsystem when both would address the issue.

Multiple production files may be modified only when they are directly required
by the same improvement. Breadth is not evidence of quality.

Before finishing:

- inspect the final diff;
- confirm the changed code is reachable from the live runtime path;
- run the most relevant available import or test command;
- leave the repository with a non-empty source-code diff.
"""
```

## 必须删除

删除或改写以下表述：

```text
Implement the diagnosis fully enough
address this class of failures
do not stop at a cosmetic one-line change
multiple related files are allowed when needed
```

尤其删除：

```text
do not stop at a one-line change
```

因为它会对模型形成“修改越大越认真”的错误激励。

## 不要加入严格 80 行上限

`MAX_PATCH_LINES = 80` 不应直接用于 evolution hard rejection。

DGM/HGM 是 open-ended self-modification，真正有价值的工具或 workflow 改动可能超过 80 行。

建议只对异常 destructive edit 做机械检查，例如：

- 单文件删除超过原文件 70%；
- 删除超过 200 行但新增很少；
- 创建 `.backup`；
- 整个文件被替换成明显不完整内容。

Patch 大小可记录到 mutation manifest，用于分析，但不要作为能力好坏的替代指标。

---

# 14. P1-8：Evidence-conditioned Code Dump

## 修改文件

```text
src/godel0/evolution/agent_code_dump.py
src/godel0/evolution/hgm_diagnose.py
```

当前 proposer code dump 可能包含：

```text
proposer/
swesmith/
utils/
```

并达到 200k chars。

改为聚焦文件选择。

## 14.1 Proposer 默认文件

始终包含：

```text
proposer/proposer_main.py
proposer/schemas.py
```

根据 evidence 动态加入：

- candidate 所属 workflow；
- operator 文件；
- stack trace 中出现的 agent-side 文件；
- mutation patch construction helper；
- statement generation helper；
- causal isolation helper。

建议最多：

```yaml
diagnosis:
  max_code_files: 12
  code_dump_clip_chars: 80000
```

## 14.2 Solver 默认文件

始终包含：

```text
coding_agent.py
llm_withtools.py
```

动态加入：

- trajectory 实际调用的 tool；
- trajectory 涉及的 agent-side helper；
- 只有 model routing/context failure 时才加入 `llm.py`；
- 只有 prompt 文件实际在 live path 使用时才加入相应 prompt。

## 14.3 不要 dump

不要加入：

```text
self-improvement prompt 本身
trusted controller
archive / tree selection
evaluation implementation
unrelated tests
所有 utils
整个 proposer 目录
整个 swesmith 目录
```

---

# 15. P1-9：Solver Diagnosis 必须基于 Intermediate Commit

## 当前问题

当前 `_build_dual_hgm_child()` 在 self-edit 之前就同时生成：

```text
proposer_diagnosis
solver_diagnosis
```

然后交给 `build_dual()`。

这意味着 solver diagnosis 看到的是 parent code，而不是 proposer edit 后的 intermediate code。

## 推荐重构

将 dual orchestration 迁移为 staged callback。

一种实现方式：

```python
ChildBuilder.build_dual(
    parent=parent,
    proposer_diagnosis=...,
    solver_diagnosis_factory=...,
)
```

流程：

```python
proposer edit
→ phase gate
→ intermediate commit

solver_diagnosis = solver_diagnosis_factory(
    worktree=worktree,
    intermediate_commit=phase_base,
)

solver edit
```

更清晰的方式是由 orchestrator 控制两个阶段：

```python
proposer_phase_result = child_builder.build_proposer_phase(...)
solver_diagnosis = diagnoser.diagnose_solver(
    agent_repo=proposer_phase_result.worktree,
    ...
)
final_result = child_builder.build_solver_phase(...)
```

但要注意 worktree 生命周期。

## 最低成本替代方案

如果暂时不重构 diagnosis 时序，则必须严格保证：

```text
proposer phase 只能修改 proposer/ 和 swesmith/
solver phase 只能修改 solver paths
```

这样 solver diagnosis 对 solver code 仍然有效。

本次建议先完成严格 role isolation，再在下一次重构中移动 solver diagnosis 时序。

---

# 16. P2-1：避免重复选择同一个 Failure Entry

## 修改文件

```text
src/godel0/evolution/entry_selector.py
src/godel0/controller/orchestrator.py
src/godel0/schemas/
src/godel0/storage/
```

## 增加 MutationAttemptRecord

建议新增：

```python
class MutationAttemptRecord(BaseModel):
    attempt_id: str
    parent_node_id: str

    proposer_entry_id: str | None = None
    solver_entry_id: str | None = None

    proposer_diagnosis_succeeded: bool = False
    solver_diagnosis_succeeded: bool = False

    proposer_patch_created: bool = False
    solver_patch_created: bool = False

    failure_stage: str | None = None
    failure_reasons: list[str] = []

    created_child_node_id: str | None = None
```

保存路径：

```text
runs/<run_id>/mutation_attempts/<attempt_id>.json
```

或：

```text
runs/<run_id>/parents/<parent_id>/mutation_attempts.jsonl
```

## Entry 选择

```python
def choose_least_attempted_failure(
    failures,
    attempt_counts: dict[str, int],
    rng: random.Random,
):
    if not failures:
        return None

    minimum = min(
        attempt_counts.get(failure.id, 0)
        for failure in failures
    )

    candidates = [
        failure
        for failure in failures
        if attempt_counts.get(failure.id, 0) == minimum
    ]

    return rng.choice(candidates)
```

优先规则：

```text
未尝试 entry
→ 尝试次数最少 entry
→ 同次数随机
```

不要永久禁止重复 entry，因为同一个 failure 可能通过不同 mutation 解决。

---

# 17. P2-2：将 Mutation Failure 反馈给后续 Self-Edit Retry

区分两类 retry：

## 同一 phase 内 retry

用于机械失败：

- empty patch；
- syntax error；
- frozen file；
- runtime artifact；
- schema incompatible；
- import failure。

将上一尝试错误反馈给 self-edit：

```text
The previous attempt was discarded because:
- it edited proposer/request.py;
- it created a backup file;
- proposer.schemas no longer accepted the legacy payload.

Start from the clean phase base and implement the same improvement without
changing frozen protocols or generated artifacts.
```

## 跨 expansion feedback

不要把所有 previous mutation failure 塞进 diagnosis prompt。

只用于 entry selection 和实验分析。

如果同一 entry 多次出现相同 mechanical failure，可以在 diagnosis evidence 中加入简短摘要：

```text
Previous mutation attempts for this entry repeatedly modified frozen transport
schemas. The diagnosis must target evolvable proposer logic instead.
```

不要加入之前完整 patch，否则容易让模型复制失败设计。

---

# 18. Child Build 阶段允许和禁止判断的内容

## 18.1 可以拒绝

Child build 可以因以下原因拒绝：

- diagnosis 无法解析；
- proposer 或 solver phase 没有 source patch；
- 修改 frozen/trusted 文件；
- Python syntax 失败；
- import 失败；
- proposer schema 不兼容；
- runtime artifact 进入 patch；
- agent entrypoint 无法启动；
- 明显文件截断；
- 创建 backup/generated artifact；
- proposer phase 修改 solver-only 文件；
- solver phase 修改 proposer-only 文件。

## 18.2 不应拒绝

不要因为以下原因在 child build 阶段拒绝：

- 修改看起来不够聪明；
- patch 比较大；
- 可能降低 benchmark 成绩；
- prompt 修改看起来太简单；
- 没有证明一定能解决 failure；
- 修改与人工预期不同；
- child 可能退化。

这些必须由：

```text
Level 1 retention
Proposer generation quality
Level 2 solver performance
HGM utility
Thompson Sampling
```

决定。

---

# 19. 机械 Gate 建议

## Proposer phase

至少运行：

```bash
python -B -m proposer.proposer_main --help
```

并验证：

```text
proposer/schemas.py import
legacy schema payload compatibility
proposer/request.py 未修改
没有直接写 TaskStore
没有读取 trusted private inputs
```

## Solver phase

至少运行：

```bash
python -B coding_agent.py --help
python -B -c "import coding_agent"
python -B -c "import llm_withtools"
```

如 agent tests 存在，可运行，但 tests 不允许被 evolution 修改。

## Final child

运行：

```bash
scripts/validate_agent_codebase.py --code-dir <worktree>
python -B -m proposer.proposer_main --help
python -B -c "import coding_agent"
```

---

# 20. 测试计划

## 20.1 Runtime artifact tests

新增测试：

```text
tracked pyc 被修改后，restore_runtime_artifacts 恢复；
untracked pycache 被删除；
normal source changes 保留；
diff_vs_commit 不包含 runtime artifacts。
```

## 20.2 Role-specific PatchGuard tests

测试：

```text
proposer 可修改 proposer/schemas.py；
proposer 不可修改 proposer/request.py；
proposer 不可修改 coding_agent.py；
solver 可修改 coding_agent.py；
solver 不可修改 proposer/；
tests/ 不可修改。
```

## 20.3 Schema compatibility tests

测试：

```text
新增 optional schema field：通过；
删除 BugGenerationPlan：失败；
删除 target_file：失败；
将 optional field 改为 required：旧 payload 失败；
旧 payload 保持可解析：通过。
```

## 20.4 Diagnosis tests

测试：

```text
包含一个 root cause：通过；
包含多个 alternative improvements：解析失败或重试；
edit_scope > 4：失败；
建议 passed=True on exception：失败；
建议编辑 proposer/request.py：失败；
diagnosis 连续解析失败：返回 None，不生成 fallback。
```

## 20.5 Dual phase tests

测试：

```text
proposer phase empty：整个 mutation 失败，不运行 solver；
proposer phase frozen-file violation：失败，不运行 solver；
proposer phase valid：创建 intermediate commit；
solver phase empty：整个 mutation 失败；
两个 phase valid：final patch 同时包含 proposer 和 solver 修改。
```

## 20.6 Entry selection tests

测试：

```text
优先未尝试 entry；
所有 entry 都尝试后选择尝试次数最少者；
同次数使用 seeded RNG；
mutation attempt 被持久化。
```

---

# 21. 推荐实施顺序

Cursor 应严格按照以下顺序修改，每一阶段完成后运行测试。

## Step 1：Runtime cleanup

修改：

```text
repository.py
self_edit.py
child_builder.py
.gitignore
```

验收：

```text
一次 self-edit import 多个 Python 模块后，
final patch 不包含任何 pyc / pycache。
```

## Step 2：Role-specific gates

修改：

```text
constants.py
patch_guard.py
child_builder.py
```

验收：

```text
proposer phase 的非法修改在 solver phase 前被拒绝。
```

## Step 3：Schema 解冻与 compatibility gate

修改：

```text
constants.py
gates.py
tests
```

验收：

```text
proposer/schemas.py 可进化；
proposer/request.py 不可进化；
旧 schema payload 保持兼容。
```

## Step 4：Prompt 改造

修改：

```text
hgm_diagnose.py
self_edit.py
```

验收：

```text
diagnosis 输出一个 root cause、一个 improvement；
不再产生 bypass trusted validator 的建议；
self-edit prompt 不再鼓励大规模重构。
```

## Step 5：删除 fallback

验收：

```text
diagnosis 解析失败后 mutation 终止；
不会生成宽泛 fallback issue。
```

## Step 6：Entry attempt tracking

修改：

```text
entry_selector.py
orchestrator.py
schemas/storage
```

验收：

```text
连续 expansion 不再稳定选择同一个 failure entry。
```

## Step 7：重新运行 3 次 evolution smoke test

配置建议：

```yaml
run:
  max_nodes: 3
  max_expansions: 12

diagnosis:
  mode: hgm_dual
  max_code_files: 12
  code_dump_clip_chars: 80000

agent:
  self_evolve_max_attempts: 3
```

---

# 22. Smoke Test 必须记录的指标

每个 expansion 输出：

```text
parent_node_id
proposer_entry_id
solver_entry_id

proposer diagnosis parse status
proposer changed files
proposer added/deleted lines
proposer phase gate result

solver diagnosis parse status
solver changed files
solver added/deleted lines
solver phase gate result

child created or not
child failure stage
Level 1 result
proposer batch result
Level 2 result
final utility
```

重点观察：

1. pyc failure 是否完全消失；
2. proposer 是否仍尝试绕过 trusted gate；
3. 平均 patch size 是否显著下降；
4. 是否仍重复同一个 entry；
5. 有多少 expansion 能真正创建 child；
6. child 创建后主要失败在 Level 1、Proposer 还是 Level 2；
7. proposer 和 solver 修改是否严格 role-separated。

---

# 23. 完成标准

本次改造完成的最低标准：

- [ ] `git ls-files` 中没有 tracked runtime artifacts；
- [ ] self-edit patch 中不出现 pyc / pycache / backup；
- [ ] proposer 和 solver phase 分别独立 gate；
- [ ] `allow_empty_phase=False`；
- [ ] `proposer/schemas.py` 可修改；
- [ ] `proposer/request.py` 不可修改；
- [ ] schema compatibility gate 真正执行；
- [ ] diagnosis 只输出一个 primary root cause；
- [ ] diagnosis 不允许绕过 trusted validator；
- [ ] self-edit prompt 不鼓励“大而全”重构；
- [ ] diagnosis 失败后不生成 generic fallback；
- [ ] entry selection 优先未尝试 entry；
- [ ] mutation failure 被结构化持久化；
- [ ] 至少一次 dual mutation 能成功创建 child；
- [ ] child 的能力优劣由真实评估而非静态 prompt 判断。

---

# 24. 最终设计原则

实现过程中始终遵守以下原则。

## 原则 1：General 不等于 Large

“通用改进”表示：

```text
不硬编码当前 task；
可以泛化到相似 failure；
修改 agent 的一般行为。
```

不表示：

```text
必须建设完整框架；
必须修改多个文件；
必须修改数百行。
```

## 原则 2：Trusted Validator 是环境，不是优化对象

Proposer 的目标是：

```text
生成真正满足 validator 的候选任务。
```

不是：

```text
改变 validator 语义；
在异常时放行；
伪造 metadata；
降低检查标准。
```

## 原则 3：机械合法性和能力价值必须分开

Child build 只判断：

```text
能否运行；
协议是否兼容；
是否触碰 frozen boundary；
patch 是否真实存在。
```

真实评估判断：

```text
是否提升；
是否退化；
是否值得继续扩展。
```

## 原则 4：一次 Mutation 只解决一个主要缺陷

一次 proposer diagnosis：

```text
一个 candidate；
一个 primary rejection mechanism；
一个 coherent improvement。
```

一次 solver diagnosis：

```text
一个 failed task；
一个 primary capability defect；
一个 coherent improvement。
```

## 原则 5：Dual Evolution 是两次标准 DGM/HGM Mutation

不要把 dual evolution 理解为：

```text
一次完整系统重构，顺便覆盖 proposer 和 solver。
```

应理解为：

```text
一次 proposer mutation
+
一次 solver mutation
=
一个联合 child。
```

---

# 25. 给 Cursor 的最终执行指令

请在 `RealLcz/Godel0` 中按本文档顺序实施改造。

要求：

1. 先阅读当前相关实现和测试，不要直接重写整个模块；
2. 每个 Step 独立提交或至少保持独立 diff；
3. 每个 Step 完成后运行对应 unit tests；
4. 不修改研究目标、scoring 公式和 Thompson Sampling；
5. 不将 trusted validation 逻辑迁入 evolvable agent；
6. 不新增绕过 validation 的 fallback；
7. 不以硬性小 patch 限制替代真实 benchmark 评估；
8. 最终提供：
   - 修改文件列表；
   - 核心行为变化；
   - 新增测试列表；
   - 测试结果；
   - 尚未完成或存在风险的部分。

实施后的系统应满足：

```text
一个 proposer failure
→ 一个聚焦 proposer improvement
→ proposer phase gate
→ 一个 solver failure
→ 一个聚焦 solver improvement
→ solver phase gate
→ mechanically valid child
→ Level 1 / Proposer / Level 2 empirical evaluation
→ HGM tree selection
```
