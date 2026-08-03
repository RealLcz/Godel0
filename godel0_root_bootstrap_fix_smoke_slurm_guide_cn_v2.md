# Gödel0 当前 Root Bootstrap 失败修复与实验执行指导书

> **适用代码基线**：`RealLcz/Godel0` commit `501a0e1c22e247dd149d2310d617b0fdb18ad50c`  
> **对应失败任务**：Job `211215`、Job `211424`  
> **目标**：让 Existing-Tests 版 RepoChain 先稳定完成 Root Bootstrap，再逐步完成 `K=1 → K=3 → K=10 → 1 epoch → 20 epochs`。

---

# 1. 当前问题结论

现在的问题已经不是上一轮的 `clean_contract_failure`。

Job `211424` 已经证明 Existing Tests 路线能够：

```text
Root Bootstrap
    ↓
找到 existing passing tests
    ↓
生成 RepoChain chain plan
    ↓
生成 production mutation
    ↓
写出 CandidateArtifact
```

当前日志摘要显示：

```text
33 plans selected existing tests
9 candidate artifacts written
0 candidates survived causal ablation / trusted validation
0 accepted tasks
```

最终：

```text
Node proposer produced no result: exit=-15
```

`exit=-15` 只是最终症状。当前真实阻塞链路是：

```text
Candidate generated
    ↓
Local Causal Ablation hard filter
    ↓
return []
    ↓
Trusted CandidateValidator 看不到 candidate
    ↓
Bootstrap 一直凑不到任务
    ↓
NodeProposerRunner 一直运行
    ↓
9000 s timeout
    ↓
SIGTERM
```

因此本轮修复应该优先解决四个问题：

1. **Local Causal Ablation 不应作为 hard admission gate**；
2. **Root Bootstrap 不应在一个 proposer subprocess 中连续尝试几十个 plan**；
3. **Bootstrap candidate 当前存在 accepted + pending 双重登记问题**；
4. **Existing Tests 应从 test-file grounding 进一步升级为显式 target/control contracts，并逐步缩小 mutation whitelist**。

---

# 2. 本轮修改后的目标架构

最终推荐的 Root Bootstrap 路径：

```text
Bootstrap Capability Prior
        ↓
Real Repository Anchor
        ↓
Existing Passing Test Discovery
        ↓
Select Target Test Nodes + Compatibility Controls
        ↓
Infer Production Subgraph
        ↓
RepoChain Semantic Chain
        ↓
Generate Multi-file Production Mutation
        ↓
LOCAL causal diagnostics
        │
        │ 只记录，不丢 candidate
        ▼
CandidateArtifact
        ↓
TRUSTED CandidateValidator
        ├── Clean PASS
        ├── F2P ≥ 1
        ├── P2P ≥ 1
        ├── Reverse PASS
        ├── Safety / Relevance / Duplicate
        └── Trusted Causal Validation
        ↓
Accepted Task
```

核心原则：

```text
Evolvable Proposer = propose
Trusted Controller = judge
```

Local RepoChain 可以做诊断，但最终 Candidate Admission 只能由 trusted controller 决定。

---

# 3. P0-1：取消 Local Causal Ablation 的 Hard Filter

## 3.1 当前问题位置

文件：

```text
initial_agent/src/proposer/workflows/repo_chain/workflow.py
```

当前 `RepoChainWorkflow.generate()` 的逻辑近似：

```python
candidates = backing.generate(plan, node_code_dir, repo_spec, output_dir)

if self.require_causal_ablation:
    ablation = self.ablation_stage.run(plan, repo_spec, candidates, contracts=None)
    if not ablation.passed:
        return []

return candidates
```

这意味着已经生成并保存到磁盘的 Candidate，在 trusted controller 看到它之前就被删掉。

## 3.2 修改原则

改成：

```text
Local Causal Ablation = diagnostic
Trusted Causal Ablation = authoritative gate
```

## 3.3 推荐代码修改

在 `RepoChainWorkflowConfig` 新增：

```python
local_causal_ablation_mode: str = "diagnostic"
```

允许：

```text
diagnostic
hard_gate
```

正式主实验建议：

```yaml
local_causal_ablation_mode: diagnostic
```

修改 `RepoChainWorkflow.__init__()`：

```python
self.local_causal_ablation_mode = str(
    _cfg_get("local_causal_ablation_mode", "diagnostic")
).strip().lower()
```

修改 `generate()`：

```python
candidates = backing.generate(
    plan,
    node_code_dir,
    repo_spec,
    output_dir,
)

if self.require_causal_ablation and candidates:
    ablation = self.ablation_stage.run(
        plan,
        repo_spec,
        candidates,
        contracts=None,
    )

    # Local result is diagnostics only by default.
    for candidate in candidates:
        metadata = getattr(candidate, "generation_metadata", None)
        if isinstance(metadata, dict):
            metadata["local_causal_ablation"] = {
                "passed": bool(ablation.passed),
                "details": ablation.details,
            }

    if (
        self.local_causal_ablation_mode == "hard_gate"
        and not ablation.passed
    ):
        return []

return candidates
```

## 3.4 验收标准

原来：

```text
9 candidate artifacts on disk
0 candidates sent to trusted validation
```

修改后必须看到：

```text
candidate artifacts generated > 0
candidates_validated > 0
validation_reports > 0
```

即使所有 candidate 最终 trusted validation 失败，也必须能看到准确的：

```text
no_f2p
no_p2p
trusted_causal_ablation_failed
reverse_not_restored
...
```

而不是只看到：

```text
Node proposer produced no result
```

---

# 4. P0-2：修复 Bootstrap Candidate 的 accepted + pending 双重登记

## 4.1 当前问题

文件：

```text
initial_agent/src/proposer/runner.py
```

Root Bootstrap 分支当前近似：

```python
candidates = self._bootstrap_candidates(request)
self._write_candidates(request, candidates, [])

for cand in candidates:
    result.add_candidate(cand, accepted=True)
    result.add_pending_candidate(cand)
```

但 trusted validation 还没有发生，所以这里不能标记 `accepted=True`。

下游 `TaskBatchBuilder` 又会：

```python
candidates_to_validate = (
    list(proposer_result.accepted_candidates)
    + list(pending_candidates)
)
```

因此同一个 bootstrap candidate 可能被重复加入。

## 4.2 修改

Bootstrap 生成阶段只允许：

```python
for cand in candidates:
    result.add_pending_candidate(cand)
```

删除：

```python
result.add_candidate(cand, accepted=True)
```

语义统一：

```text
Proposer emitted candidate
→ pending_validation
→ Trusted CandidateValidator
→ accepted / rejected
```

## 4.3 验收

一个 candidate 必须满足：

```text
pending_candidates: 1
accepted_candidates before trusted validation: 0
trusted validation count: 1
```

禁止：

```text
同一个 candidate validated twice
```

---

# 5. P0-3：Root Bootstrap 改为小批量增量生成

## 5.1 当前问题

当前 Root Bootstrap 在一次 proposer subprocess 中可能连续跑大量 plan：

```text
workflow.bootstrap(... target_count=10, max_candidates=50)
```

如果 candidate survival rate 很低，就会出现：

```text
Plan 1
Plan 2
...
Plan 33
...
Plan 50
```

全部塞在同一个 proposer process 里。

而 controller 只有等 subprocess 返回之后，才能进行 trusted validation。

这导致：

```text
Generate for hours
→ no trusted feedback
→ timeout
→ result file not written
→ all work in this subprocess effectively lost
```

## 5.2 新配置

在：

```text
src/godel0/config.py
RepoChainWorkflowConfig
```

新增：

```python
bootstrap_plans_per_call: int = 2
```

建议默认：

```yaml
bootstrap_plans_per_call: 2
```

## 5.3 Bootstrap 必须支持 offset

如果每次只跑前两个 plan，但每次重新从头构造 plans，会无限重复同样的 anchor。

因此 `RepoChainWorkflow.bootstrap()` 增加：

```python
plan_offset: int = 0
plan_limit: int | None = None
```

建议：

```python
def bootstrap(
    self,
    repo_spec,
    output_dir: str,
    capability_prior=None,
    target_count: int = 10,
    max_candidates: int | None = None,
    plan_offset: int = 0,
    plan_limit: int | None = None,
):
    ...
```

构造 plan 时：

```python
chunk_size = max(1, int(plan_limit or self.bootstrap_plans_per_call))
needed = plan_offset + chunk_size

all_plans = build_bootstrap_plans(
    prior,
    repo_spec,
    target_count=target_count,
    max_plans=needed,
    code_locator=self.code_locator,
)

plans = all_plans[plan_offset : plan_offset + chunk_size]
```

然后只执行这个 chunk。

## 5.4 在 `_bootstrap_candidates()` 中传入 generation_attempt

文件：

```text
initial_agent/src/proposer/runner.py
```

使用：

```python
plans_per_call = int(
    (request.workflow_config or {}).get(
        "bootstrap_plans_per_call",
        2,
    )
)

plan_offset = int(request.generation_attempt or 0) * plans_per_call
```

调用：

```python
candidates = workflow.bootstrap(
    repo_spec=repo_spec,
    output_dir=cand_dir,
    capability_prior=BOOTSTRAP_CAPABILITY_PRIOR,
    target_count=min(
        int(request.target_batch_size or 1),
        plans_per_call,
    ),
    max_candidates=plans_per_call,
    plan_offset=plan_offset,
    plan_limit=plans_per_call,
)
```

## 5.5 关键：必须把 attempted plans 回传给 TaskBatchBuilder

当前 `TaskBatchBuilder` 只有在：

```python
generated_this_attempt > 0
```

时才继续下一轮。

因此即使本 chunk 的 2 个 plan 都没产出 candidate，也必须记录：

```text
plans_attempted = 2
```

建议让 bootstrap 返回：

```python
BootstrapChunkResult(
    candidates=[...],
    plans=[...],
)
```

或者最小改法：

```python
candidates, plans = workflow.bootstrap(...)
result.plans = [p.model_dump() for p in plans]
```

这样下游已有逻辑：

```python
generated_this_attempt = max(
    generated_this_attempt,
    len(proposer_result.plans),
)
```

会正确消耗 generation budget，并继续下一次 chunk。

## 5.6 理想执行节奏

修改后 Root Bootstrap：

```text
Proposer Call 0
→ Plan 0-1
→ emit 0~2 candidates
→ return immediately

Trusted Validation
→ feedback written

Proposer Call 1
→ Plan 2-3
→ emit 0~2 candidates
→ return

Trusted Validation
→ feedback written

...

直到 accepted tasks == K
或 generation budget exhausted
```

这比当前 2.5 小时单进程安全得多。

---

# 6. P0-4：Proposer Timeout 不再使用 candidate_timeout × 50

## 6.1 当前逻辑

当前 orchestrator：

```python
timeout_sec = (
    config.proposer.candidate_timeout_sec
    * config.tasks.max_generation_candidates
)
```

当前配置：

```text
180 × 50 = 9000 s = 2.5 hours
```

这正好对应 Job 211424 最后的 SIGTERM 时间。

## 6.2 修改 Config

在：

```text
src/godel0/config.py
ProposerConfig
```

新增：

```python
batch_timeout_sec: int = 900
```

建议：

```text
candidate_timeout_sec = 180
batch_timeout_sec = 900
bootstrap_plans_per_call = 2
```

## 6.3 修改 Orchestrator

原：

```python
NodeProposerRunner(
    ...,
    timeout_sec=(
        config.proposer.candidate_timeout_sec
        * config.tasks.max_generation_candidates
    ),
)
```

改：

```python
NodeProposerRunner(
    ...,
    timeout_sec=config.proposer.batch_timeout_sec,
)
```

这样每个 proposer chunk 最长 15 分钟，而不是整轮 2.5 小时。

---

# 7. P0-5：加入 Bootstrap-Only 调试模式

为了避免每次验证 Root Bootstrap 都意外继续进入 evolution，建议加一个正式 debug 开关。

## 7.1 Config

`RunConfig` 增加：

```python
bootstrap_only: bool = False
```

## 7.2 Orchestrator

在：

```python
if not self._ensure_root_bootstrap():
    ...
```

成功之后、正式 evolution loop 之前增加：

```python
if getattr(self.config.run, "bootstrap_only", False):
    logger.info("Bootstrap-only mode completed successfully")
    return
```

这样 smoke test 可以明确测试：

```text
Root task generation
+
Trusted validation
+
Root Level2（是否执行取决于放置位置）
```

推荐把 `bootstrap_only` return 放在 Root Level2 完成之后。

本指导书推荐：

```text
bootstrap_only = Root Bootstrap + Root Level2
```

因为这才是完整的 Epoch-0 初始化路径。

---

# 8. P1-1：Existing Tests 从 Test File 升级为 Exact Test Node IDs

当前 `_select_passing_existing_tests()` 返回类似：

```text
test/units/vars/test_manager.py
```

建议升级为：

```text
test/units/vars/test_manager.py::TestVariableManager::test_precedence
```

原因：

1. test file 可能包含几十个测试；
2. mutation agent 不知道哪个行为是 target；
3. trusted validation 的 F2P/P2P 太粗；
4. causal ablation 可能因为大量无关 tests 产生噪声。

## 8.1 Discovery

先运行：

```bash
pytest -q --collect-only <test_file>
```

或直接 clean run：

```bash
pytest -v <test_file>
```

从输出解析完整 node ID。

Existing test contract 数据结构建议：

```python
@dataclass
class ExistingTestContract:
    node_id: str
    file_path: str
    role: str  # target | control
    clean_passed: bool
    related_production_files: list[str]
```

---

# 9. P1-2：显式区分 F2P Target 和 P2P Control

当前 existing-test taxonomy 近似：

```python
return {
    "FAIL_TO_PASS": nodeids,
    "PASS_TO_PASS": [],
}
```

建议改为 contract plan 明确输出：

```json
{
  "existing_f2p_targets": [
    "T1",
    "T2"
  ],
  "existing_p2p_controls": [
    "T3"
  ]
}
```

其中 `T1/T2/T3` 是系统提供的 whitelist ID，不能由 LLM 自由写路径。

例如：

```text
T1 = test/...::test_variable_precedence
T2 = test/...::test_hostvars
T3 = test/...::test_empty_inventory
```

目标：

```text
Clean:
T1 PASS
T2 PASS
T3 PASS

Bugged:
T1 or T2 FAIL
T3 PASS

Reverse:
T1/T2 PASS
```

Trusted Validator 仍然必须根据真实执行结果重新计算 F2P/P2P，绝不能相信 Proposer 声称的标签。

---

# 10. P1-3：Existing Test → Production Subgraph，再生成 mutation

当前流程仍比较接近：

```text
anchor
→ broad production context
→ nearby tests
→ mutate allowed production context
```

建议逐步改成：

```text
selected exact test nodes
→ imports / symbol references
→ related production files
→ one-hop production imports
→ mutation whitelist
```

例如：

```text
test_variable_precedence
    ↓
imports ansible.vars.manager
    ↓
lib/ansible/vars/manager.py
    ↓
imports / calls inventory manager
    ↓
lib/ansible/inventory/manager.py
```

最终 LLM 只能从这个 subgraph 的 AST symbol IDs 中选择 mutation sites。

不要再把整个 `context_file_budget=10` 都当作同等可 mutation 区域。

---

# 11. P1-4：Symbol 使用稳定 ID，而不是自由文本

当前仍可能出现：

```text
mutation symbol not found
```

建议 symbol catalog：

```json
{
  "S17": {
    "file": "lib/ansible/vars/manager.py",
    "qualified_name": "VariableManager.get_vars"
  },
  "S18": {
    "file": "lib/ansible/inventory/manager.py",
    "qualified_name": "InventoryManager.parse_sources"
  }
}
```

Prompt 只允许输出：

```json
{
  "mutation_site_ids": ["S17", "S18"]
}
```

Trusted materializer 再解析：

```text
S17 → exact file + symbol
```

同时天然避免：

```text
symbol hallucination
file/symbol mismatch
```

---

# 12. P1-5：取消 Causal Ablation 的 Admission Hard Gate

这一版建议进一步简化：

> **不要再用任何 causal ablation 条件决定 Candidate 是否有效。**

包括当前的：

```text
完整 Bug = ΔA + ΔB + ΔC

修掉 ΔA，只剩 ΔB+ΔC → 必须 FAIL
修掉 ΔB，只剩 ΔA+ΔC → 必须 FAIL
修掉 ΔC，只剩 ΔA+ΔB → 必须 FAIL
```

全部取消 hard gate。

原因是这个条件并不能可靠证明“真正的 multi-file causal chain”，反而会误杀很多合理的 repository-level bug。

例如一个真实的跨文件传播链：

```text
A: producer
↓
B: transformer
↓
C: consumer
```

如果 A 是 dominant root cause，那么：

```text
修复 A
→ 下游错误状态不再传播
→ 整体测试恢复 PASS
```

这完全可能是一个合理的 multi-file bug，但当前 leave-one-out gate 会直接 reject。

反过来，一个：

```text
A = 真 bug
B = 另一个独立真 bug
C = 完全无关 filler
```

也可能满足：

```text
修复任意一个文件后仍 FAIL
```

因此这条规则既会误杀合理任务，也不能充分排除 filler。

## 12.1 第一版真正的 Hard Admission 条件

建议 Candidate Admission 只要求：

```text
1. Clean Repo 上 target tests PASS
2. Full Bug 后至少一个 target test FAIL
3. Reverse Full Bug 后 target tests 恢复 PASS
4. 至少一个 F2P
5. 至少一个 P2P
6. production changed files >= 2
7. mutation sites >= 2
8. patch safety PASS
9. no leakage PASS
```

即：

```text
ValidTask =
CleanPass
∧ FullBugFail
∧ ReversePass
∧ F2P
∧ P2P
∧ MultiFile
∧ Safety
∧ NoLeakage
```

这是第一阶段最推荐的正式 admission 定义。

## 12.2 Causal Analysis 全部降级为 Soft Metrics

仍然可以运行：

```text
Leave-One-Out:
Full Bug - ΔA
Full Bug - ΔB
Full Bug - ΔC
```

以及：

```text
Isolated Mutation:
Clean + ΔA
Clean + ΔB
Clean + ΔC
```

但只记录：

```json
{
  "causal_analysis": {
    "leave_one_out": {
      "A.py": "PASS",
      "B.py": "FAIL",
      "C.py": "FAIL"
    },
    "isolated": {
      "A.py": ["T1"],
      "B.py": [],
      "C.py": []
    },
    "independently_active_file_count": 1,
    "active_file_ratio": 0.33
  }
}
```

这些数据用于：

```text
task quality analysis
debugging
ablation study
future stricter admission experiments
```

但第一阶段不阻止 Candidate 进入 Trusted Validation 和 Level 2。

## 12.3 允许的三类 Multi-file Bug

RepoChain 不应该只允许“每个文件单独都能致错”的任务。

至少允许：

### Type I：Independent Multi-site

```text
ΔA only → FAIL
ΔB only → FAIL
```

### Type II：Interaction Bug

```text
ΔA only → PASS
ΔB only → PASS
ΔA + ΔB → FAIL
```

### Type III：Dominant Root Cause + Propagation

```text
ΔA only → FAIL
ΔB only → PASS
Full Bug → FAIL

repair A from Full Bug → PASS
```

这三类都可能需要 repository-level reasoning。

真正需要防的是：

```text
Type IV：Filler

一个单文件真 bug
+ 若干与 target behavior 无关的无意义修改
```

第一阶段先通过 semantic chain、test-to-production dependency、patch relevance 和 multi-file requirement 控制；不要再通过 leave-one-out hard gate 强行过滤。

## 12.4 代码层修改

Local RepoChain：

```python
if self.require_causal_ablation and candidates:
    ablation = self.ablation_stage.run(...)

    for candidate in candidates:
        candidate.generation_metadata["causal_analysis"] = {
            "passed_under_old_rule": bool(ablation.passed),
            "details": ablation.details,
        }

# 永远不要因为 causal analysis return []
return candidates
```

Trusted `CandidateValidator`：

```python
# 仍然可以执行 causal analysis
causal_report = self._causal_ablation(...)

# 只写入 validation report
report.causal_analysis = causal_report

# 第一阶段不因为 causal_report 失败而：
# report.passed = False
```

正式 admission 仅由：

```text
Clean / F2P / P2P / Reverse / Multi-file / Safety / NoLeakage
```

决定。

等系统稳定跑完 20 epochs 后，再增加 ablation：

```text
Gödel0-NoCausalGate
Gödel0-StrictCausalGate
```

比较 task yield、task difficulty 和最终外部 benchmark 表现，再决定 strict causal gate 是否值得保留。

---

# 13. 建议新增的配置字段

`src/godel0/config.py`：

```python
@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    run_name: str | None = None
    max_nodes: int = 200
    max_expansions: int = 200
    resume_from: str | None = None
    bootstrap_only: bool = False


@dataclass(frozen=True)
class RepoChainWorkflowConfig:
    min_files: int = 2
    max_files: int = 6
    min_mutation_sites: int = 3
    max_mutation_sites: int = 8
    context_file_budget: int = 10

    require_generated_contracts: bool = False

    # Local = evolvable proposer diagnostic only.
    require_causal_ablation: bool = True
    local_causal_ablation_mode: str = "diagnostic"

    bootstrap_plans_per_call: int = 2

    mutation_operator: str = "trajectory_conditioned_chain_mutation"


@dataclass(frozen=True)
class ProposerConfig:
    initial_workflow: str = "repo_chain"
    repo_chain: RepoChainWorkflowConfig = field(
        default_factory=RepoChainWorkflowConfig
    )

    candidate_timeout_sec: int = 180
    batch_timeout_sec: int = 900

    max_patch_lines: int = 80
    forbid_test_file_edits: bool = True
    require_f2p: bool = True
    contract_test_renderer: str = ""
    allow_human_curated_data: bool = False
    allow_workflow_fallback: bool = False
```

第一版不建议再增加新的 trusted causal hard-gate 阈值。

建议只增加：

```python
record_causal_diagnostics: bool = True
causal_ablation_hard_gate: bool = False
```

正式第一阶段：

```yaml
record_causal_diagnostics: true
causal_ablation_hard_gate: false
```

等 20 epochs 跑通后，再把：

```yaml
causal_ablation_hard_gate: true
```

作为单独 ablation 实验。

---

# 14. 第一阶段 Unit Tests

修改代码后先不要提交 Slurm。

运行：

```bash
export PYTHONPATH="$PWD:$PWD/src"
pytest -q
```

至少新增以下测试。

## 14.1 Local causal failure 不删除 candidate

```text
test_repo_chain_local_causal_is_diagnostic.py
```

输入：

```text
backing generator emits 1 candidate
local causal result = failed
mode = diagnostic
```

期望：

```text
len(workflow.generate(...)) == 1
```

而不是 `0`。

## 14.2 hard_gate ablation 仍可工作

```text
mode = hard_gate
local causal = failed
```

期望：

```text
len(candidates) == 0
```

用于 ablation experiment。

## 14.3 Bootstrap candidate 只进入 pending

期望：

```text
accepted_candidates == []
pending_candidates == [candidate]
```

## 14.4 Bootstrap chunk offset

调用：

```text
generation_attempt=0 → plans 0-1
generation_attempt=1 → plans 2-3
```

期望 anchor / plan 不重复。

## 14.5 Zero-yield chunk 仍继续消耗 plan budget

第一个 chunk：

```text
2 plans
0 candidate
```

期望：

```text
generated_this_attempt == 2
TaskBatchBuilder 进入下一 attempt
```

禁止：

```text
proposer_generated_zero_candidates → immediate break
```

## 14.6 Trusted validator 能看到 local-causal-failed candidate

输入一个：

```text
local causal = failed
```

但 candidate artifact 有合法 patch。

期望：

```text
validator.validate called exactly once
```

---

# 15. 第二阶段：Root Bootstrap K=1 冒烟配置

新建：

```text
configs/smoke_ansible_root_k1.yaml
```

建议：

```yaml
run:
  seed: 7302
  run_name: "ansible_root_k1_smoke"
  max_nodes: 1
  max_expansions: 5
  bootstrap_only: true

models:
  solver_model: "Qwen/Qwen3.6-35B-A3B"
  proposer_model: "Qwen/Qwen3.6-35B-A3B"
  diagnose_model: "Qwen/Qwen3.6-35B-A3B"
  self_improve_model: "Qwen/Qwen3.6-35B-A3B"
  temperature: 0.0
  max_tokens: 8192

agent:
  max_steps: 100
  max_tool_errors: 5
  trajectory_format: "jsonl"
  self_evolve_timeout_sec: 1800

tasks:
  batch_size: 1
  max_generation_candidates: 5
  candidates_per_signature: 1
  allow_same_repo_transfer: true
  allow_cross_repo_transfer: false

scoring:
  regression_threshold: 0.8
  regression_weight: 0.5
  proposer_target_accuracy: 0.5
  min_parent_solved_tasks: 1
  mode: hgm
  selection:
    strategy: thompson_sampling
    num_pseudo_descendant_evals: 10

evaluation:
  solver_rollouts: 1
  deterministic: true
  level1_timeout_sec: 1200
  level2_timeout_sec: 1200
  max_workers: 1

proposer:
  initial_workflow: repo_chain
  candidate_timeout_sec: 180
  batch_timeout_sec: 900
  repo_chain:
    min_files: 2
    max_files: 4
    min_mutation_sites: 2
    max_mutation_sites: 5
    context_file_budget: 8

    require_generated_contracts: false

    # Local gate OFF; diagnostics still recorded.
    require_causal_ablation: true
    local_causal_ablation_mode: diagnostic

    bootstrap_plans_per_call: 2
    mutation_operator: "trajectory_conditioned_chain_mutation"

execution:
  backend: "subprocess"
  scratch_root: "./scratch_ansible_root_k1_smoke"
  clean_env: true
  network_disabled: true

paths:
  agent_repo: "./agent_repo_ansible_evolve20_pass1"
  repo_pool: "./repo_pool"
  runs: "./runs_ansible_root_k1_smoke"
  task_store: "./task_store_ansible_root_k1_smoke"
```

> 第一次 K=1 smoke 的目标是验证 candidate 能进入 trusted validation。  
> 如果当前 trusted causal 仍使用旧的严格 gate，可在这个 smoke 中临时关闭 **trusted** causal gate；local causal 已经只是 diagnostic。

---

# 16. K=1 Smoke 的成功判据

必须至少看到：

```text
Root bootstrap started
Existing passing tests selected
CandidateArtifact generated
CandidateValidator.validate called
```

理想结果：

```text
accepted_tasks = 1
root bootstrap complete
root Level2 starts
root Level2 completes
bootstrap-only mode exits normally
```

最低可接受结果：

```text
candidate reaches trusted validation
```

即使 trusted validation reject，也必须能得到具体 rejection：

```text
no_f2p
no_p2p
reverse_not_restored
trusted_causal_ablation_failed
```

不能再出现：

```text
9 candidate artifacts on disk
0 validation_reports
```

---

# 17. 第三阶段：K=1 完整 Trusted Validation Smoke

K=1 的目标不是重新打开 causal hard gate，而是确认完整 trusted validation 可以稳定接受任务。

保持：

```yaml
causal_ablation_hard_gate: false
record_causal_diagnostics: true
```

目标：

```text
至少 1 candidate:
Clean PASS
→ Full Bug FAIL
→ F2P ≥ 1
→ P2P ≥ 1
→ Reverse PASS
→ Multi-file PASS
→ Safety PASS
→ NoLeakage PASS
→ Accepted
```

同时 causal analysis 正常写入 report：

```text
leave_one_out
isolated_mutations
active_file_count
active_file_ratio
```

但不参与 admission。

这一步成功后直接扩大到 K=3。

---

# 18. 第四阶段：K=3 Preflight

复制 K=1 config：

```text
configs/smoke_ansible_root_k3.yaml
```

修改：

```yaml
tasks:
  batch_size: 3
  max_generation_candidates: 15
```

仍然：

```yaml
run:
  bootstrap_only: true
```

成功标准：

```text
accepted_tasks = 3
root Level2 outcomes = 3
normal exit
```

---

# 19. 第五阶段：K=10 Root Bootstrap

新建：

```text
configs/smoke_ansible_root_k10.yaml
```

修改：

```yaml
run:
  bootstrap_only: true

tasks:
  batch_size: 10
  max_generation_candidates: 50
```

仍然使用：

```yaml
bootstrap_plans_per_call: 2
batch_timeout_sec: 900
```

这里 `max_generation_candidates=50` 是**整轮 Root Bootstrap budget**，不是一个 subprocess 一次跑 50 个 plan。

正确行为：

```text
call 0 → 2 plans → validate
call 1 → 2 plans → validate
call 2 → 2 plans → validate
...
直到 10 accepted tasks
```

---

# 20. 第六阶段：1 Complete Evolution Epoch

K=10 Root 成功后，新建：

```text
configs/smoke_ansible_one_epoch.yaml
```

修改：

```yaml
run:
  bootstrap_only: false
  max_nodes: 1
  max_expansions: 10
```

目标完整链路：

```text
Root Bootstrap K=10
↓
Root Level2 Pass@1
↓
Thompson Sampling selects Root
↓
Joint Diagnosis
↓
Self Edit
↓
Child
↓
Level1 Regression
↓
Child Proposer
↓
Existing-Test RepoChain K=10
↓
Trusted Validation
↓
Child Level2
↓
Child COMPLETE
↓
nodes_created = 1
```

只有这一步成功后，才提交 20 epochs。

---

# 21. 正式 20-Epoch Config

当前 `evolve20_ansible_pass1_existing10.yaml` 实际：

```yaml
max_nodes: 10
```

因此它并不是 20 epochs。

正式配置应新建：

```text
configs/evolve20_ansible_existing20.yaml
```

核心：

```yaml
run:
  seed: 7302
  run_name: "ansible_evolve20_existing20"
  max_nodes: 20
  max_expansions: 200
  bootstrap_only: false

tasks:
  batch_size: 10
  max_generation_candidates: 50
  candidates_per_signature: 5
  allow_same_repo_transfer: true
  allow_cross_repo_transfer: false

proposer:
  candidate_timeout_sec: 180
  batch_timeout_sec: 900
  repo_chain:
    min_files: 2
    max_files: 6
    min_mutation_sites: 3
    max_mutation_sites: 8
    context_file_budget: 10

    require_generated_contracts: false
    require_causal_ablation: true
    local_causal_ablation_mode: diagnostic

    bootstrap_plans_per_call: 2
    mutation_operator: "trajectory_conditioned_chain_mutation"

evaluation:
  solver_rollouts: 1
  deterministic: true
  level1_timeout_sec: 1200
  level2_timeout_sec: 1200
  max_workers: 1
```

Formal run 中 trusted causal 应开启。

---

# 22. 本地 / 交互节点运行命令

CLI 当前支持：

```bash
python -m godel0.cli run --config <config>
```

建议：

```bash
cd /path/to/Godel0
export PYTHONPATH="$PWD:$PWD/src"
export VLLM_HOST=127.0.0.1
export VLLM_PORT=8000

python -m godel0.cli run \
  --config configs/smoke_ansible_root_k1.yaml
```

Qwen 模型通过：

```text
http://${VLLM_HOST}:${VLLM_PORT}/v1
```

访问 vLLM。

---

# 23. Slurm：推荐统一提交脚本

建议新增：

```text
scripts/slurm/run_godel0_vllm.sbatch
```

下面是可直接改成你集群参数的模板。

> `--partition`、`--account`、Conda 路径需要替换为你集群已有值。  
> 不需要 Docker。

```bash
#!/bin/bash
#SBATCH --job-name=godel0-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
# SBATCH --partition=<YOUR_PARTITION>
# SBATCH --account=<YOUR_ACCOUNT>

set -euo pipefail

CONFIG=${CONFIG:?"Pass CONFIG=/path/to/config.yaml to sbatch"}

ROOT=${SLURM_SUBMIT_DIR}
cd "${ROOT}"

mkdir -p logs

# 按你当前 Job 211215/211424 使用的环境修改。
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HGM

export PYTHONPATH="${ROOT}:${ROOT}/src"
export VLLM_HOST=127.0.0.1
export VLLM_PORT=8000

MODEL="Qwen/Qwen3.6-35B-A3B"
VLLM_LOG="logs/vllm_${SLURM_JOB_ID}.log"
RUN_LOG="logs/godel0_${SLURM_JOB_ID}.log"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[$(date)] Starting ${MODEL} with TP=8"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size 8 \
  > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

# 最多等待 15 分钟。
READY=0
for i in $(seq 1 180); do
  if curl -fsS \
    "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" \
    >/dev/null 2>&1; then
    READY=1
    break
  fi

  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready"
    tail -n 200 "${VLLM_LOG}" || true
    exit 1
  fi

  sleep 5
done

if [[ "${READY}" != "1" ]]; then
  echo "vLLM readiness timeout"
  tail -n 200 "${VLLM_LOG}" || true
  exit 1
fi

echo "[$(date)] vLLM ready"
echo "[$(date)] Running config: ${CONFIG}"

python -m godel0.cli run \
  --config "${CONFIG}" \
  2>&1 | tee "${RUN_LOG}"

echo "[$(date)] Godel0 run completed"
```

---

# 24. Slurm 冒烟测试提交顺序

严格按以下顺序。

## 24.1 K=1 Root Bootstrap

```bash
sbatch \
  --job-name=godel0-k1 \
  --export=ALL,CONFIG=configs/smoke_ansible_root_k1.yaml \
  scripts/slurm/run_godel0_vllm.sbatch
```

查看：

```bash
squeue -u $USER
```

完成后：

```bash
sacct -j <JOB_ID> \
  --format=JobID,JobName,State,Elapsed,ExitCode
```

日志：

```bash
tail -f logs/slurm_godel0-k1_<JOB_ID>.out
```

## 24.2 K=1 完整 Trusted Validation

```bash
sbatch \
  --job-name=godel0-k1v \
  --export=ALL,CONFIG=configs/smoke_ansible_root_k1_validated.yaml \
  scripts/slurm/run_godel0_vllm.sbatch
```

该配置保持：

```yaml
causal_ablation_hard_gate: false
record_causal_diagnostics: true
```

目标是验证：

```text
Clean
F2P
P2P
Reverse
Multi-file
Safety
NoLeakage
```

而不是重新要求 leave-one-out 必须通过。

## 24.3 K=3

```bash
sbatch \
  --job-name=godel0-k3 \
  --export=ALL,CONFIG=configs/smoke_ansible_root_k3.yaml \
  scripts/slurm/run_godel0_vllm.sbatch
```

## 24.4 K=10 Root

```bash
sbatch \
  --job-name=godel0-k10 \
  --export=ALL,CONFIG=configs/smoke_ansible_root_k10.yaml \
  scripts/slurm/run_godel0_vllm.sbatch
```

## 24.5 1 Epoch

```bash
sbatch \
  --job-name=godel0-e1 \
  --export=ALL,CONFIG=configs/smoke_ansible_one_epoch.yaml \
  scripts/slurm/run_godel0_vllm.sbatch
```

## 24.6 20 Epoch Formal

只有前五步都成功后：

```bash
sbatch \
  --job-name=godel0-e20 \
  --export=ALL,CONFIG=configs/evolve20_ansible_existing20.yaml \
  scripts/slurm/run_godel0_vllm.sbatch
```

---

# 25. 每个 Slurm Job 完成后必须检查什么

不要只看：

```text
COMPLETED / FAILED
```

必须检查 pipeline counters。

重点检查：

```text
plans attempted
candidates generated
candidates validated
accepted tasks
rejection reasons
trusted causal stats
root Level2 outcomes
nodes created
expansions attempted
```

建议 grep：

```bash
grep -R "accepted" runs* | head -n 50 || true
grep -R "no_f2p" runs* | head -n 50 || true
grep -R "no_p2p" runs* | head -n 50 || true
grep -R "trusted_causal" runs* | head -n 50 || true
grep -R "Node proposer produced no result" runs* || true
```

---

# 26. 每一阶段的 Go / No-Go 标准

## Stage A：Unit Tests

Go：

```text
all tests pass
```

## Stage B：K=1 local causal diagnostic

Go：

```text
candidate reaches trusted validator
validation_reports > 0
```

## Stage C：K=1 no trusted causal

Go：

```text
accepted_tasks >= 1
```

## Stage D：K=1 完整 Trusted Validation

Go：

```text
accepted_tasks >= 1
Clean/F2P/P2P/Reverse/Multi-file/Safety/NoLeakage 全部通过
causal diagnostics 已成功记录
```

## Stage E：K=3

Go：

```text
accepted_tasks = 3
root Level2 = 3 outcomes
```

## Stage F：K=10

Go：

```text
accepted_tasks = 10
root bootstrap completes within expected wall time
```

## Stage G：1 epoch

Go：

```text
nodes_created = 1
```

## Stage H：20 epochs

Go：

```text
nodes_created = 20
```

允许：

```text
expansions_attempted > 20
```

因为 Level1 / Proposer / Trusted Validation 失败的 expansion 不计 completed epoch。

---

# 27. 当前最重要的 Debug 统计

下一轮一定要保证每个 proposer chunk 都落盘：

```json
{
  "generation_attempt": 3,
  "plan_offset": 6,
  "plans_attempted": 2,
  "candidate_artifacts": 1,
  "sent_to_trusted_validation": 1,
  "accepted": 0,
  "rejections": {
    "no_f2p": 0,
    "no_p2p": 0,
    "trusted_causal_ablation_failed": 1
  }
}
```

不要再让最终日志只剩：

```text
Node proposer produced no result: exit=-15
```

即使超时，也应该能知道：

```text
最后完成到哪个 plan
哪些 candidate 已生成
哪些 candidate 已验证
为什么被拒绝
```

---

# 28. 推荐本轮实际提交顺序

本轮不要直接重跑 `evolve20_ansible_pass1_existing10.yaml`。

建议：

```text
1. 修改 Local Causal hard filter
2. 修 Bootstrap accepted + pending double classification
3. 实现 bootstrap chunk + generation_attempt offset
4. 修改 proposer batch timeout
5. 加 bootstrap_only
6. Unit Tests
7. Slurm K=1，causal hard gate 关闭、diagnostics 开启
8. Slurm K=1，完整 Trusted Validation
9. Slurm K=3
10. Slurm K=10 Root
11. Slurm 1 epoch
12. Slurm 20 epochs
13. 20 epochs 稳定后再做 Strict Causal Gate ablation
```

---

# 29. 最终判断

Job 211424 不应该被解读为：

```text
Existing Tests 方案失败
```

更准确的是：

```text
Generated Contract bottleneck
        ↓ 已解决
Existing Tests successfully selected
        ↓
Mutation candidates successfully generated
        ↓
Local causal hard filter removes all candidates
        ↓
Trusted validation cannot provide useful feedback
        ↓
Monolithic proposer runs until 9000 s timeout
```

因此本轮最关键的修复是：

```text
Local Causal = diagnostic
Trusted Validator = sole admission judge
```

以及：

```text
Monolithic Bootstrap
→ Incremental Generate → Validate → Feedback
```

完成这两项后，下一次 K=1 smoke 才能真正告诉我们 Remaining Bottleneck 是：

```text
F2P
P2P
Reverse
Multi-file relevance
Safety
NoLeakage
```

而 causal ablation 只作为分析元数据，不再阻止系统启动。

第一阶段的目标是先让 Gödel0 完整跑通：

```text
Root Bootstrap
→ Root Level2
→ 1 Epoch
→ 20 Epochs
```

等 pipeline 稳定后，再通过严格 causal gate 的 ablation 实验判断这类限制是否真的提升任务质量，而不是在系统尚未启动时就把它作为必要条件。
