# Gödel0 Bug 修改指导书
## 目标：HGM-style Self-Improvement + Thompson Sampling + Evolvable RepoChain Proposer + Apptainer/HPC

**审查对象**：`RealLcz/Godel0`  
**审查基线 commit**：`572b6b9914f50dbdc61bc9713a8150d5b2a0c957`  
**适用阶段**：正式 20-epoch / 大规模实验之前的 P0/P1 修复

---

# 0. 最终目标与不可破坏的不变量

本次修改不是重新设计 Gödel0，而是把当前已经基本正确的架构真正接通。

Gödel0 的最终主体应保持：

```text
HGM-style Evolution
        +
Thompson Sampling Tree Selection
        +
Evolvable ProposerTaskProvider
        +
RepoChain as the default Proposer workflow
```

主循环：

```text
Parent Node
    ↓
HGM-style Thompson Sampling selection
    ↓
Joint Cycle Diagnosis
    ↓
Self-Improvement Coding Task
    ↓
Coding Agent self-edit
    ↓
Child Node
    ↓
Level 1 Regression
    ↓
Child Proposer
    ↓
RepoChain Workflow
    ↓
Trusted-valid Coding Tasks
    ↓
Level 2 Solver Evaluation
    ↓
Update utility measurements
    ↓
Archive / Tree
```

需要始终保持以下不变量：

```text
1. 一个 Node 是一个完整 Agent System。
2. 只有一棵 Evolution Tree。
3. Solver、Proposer、RepoChain、shared tools 都属于 Node commit。
4. Self-improvement 不提前选择 Solver 或 Proposer。
5. Joint Diagnoser 只选择一个最高价值 primary root cause。
6. RepoChain 是 Proposer workflow，不是 tool，也不是普通 mutation strategy。
7. Mutation backend 是 RepoChain 内部实现机制。
8. Trusted Controller 不依赖 RepoChain 的内部实现。
9. BenchmarkTaskProvider 与 ProposerTaskProvider 对 Solver 暴露同一 Task 接口。
10. Trusted Validator、TaskStore truth、Tree Controller 不允许 self-edit。
11. Parent selection 使用 HGM-style Thompson Sampling，不使用 epsilon-greedy。
12. 正式 HPC 实验必须完全支持 Apptainer，不依赖 Docker。
```

---

# 1. 当前问题总览

| ID | 严重度 | 问题 | 直接后果 |
|---|---|---|---|
| BUG-01 | P0 | `TaskGenerationContext.level1_result` 不是 dataclass field | `_generate_batch()` 可能直接 TypeError |
| BUG-02 | P0 | RepoChainWorkflow 没有真正进入 runtime | 实际仍是 SWE-smith backend 直接生成 |
| BUG-03 | P0 | `initial_workflow: repo_chain` 是死配置 | 配置与真实行为不一致 |
| BUG-04 | P0 | Root bootstrap 传入空 capability prior | Root 生成 0 candidate |
| BUG-05 | P0 | Root bootstrap 绕过 RepoChain | 即使修空 prior，也退化成 LM Modify |
| BUG-06 | P0 | Causal ablation 没有真正 enforce | 松散单文件 bug 可能被当成 RepoChain task |
| BUG-07 | P0 | Solver patch 统计读取错误 | Solver task 可能被误判为 empty patch |
| BUG-08 | P0 | Parent/Child 5+5 quota 没有真正 enforce | task source 设计失效 |
| BUG-09 | P0 | Source provenance 分类逻辑错误 | child trajectory 容易被标成 parent failure |
| BUG-10 | P0 | Parent selection 仍是 epsilon-greedy | 偏离 HGM Thompson Sampling |
| BUG-11 | P0 | TS 所需 utility measurements 缺失 | 无法忠实实现 descendant TS |
| BUG-12 | P0 | HGM quality gate 没接到 selection | 不合格 Proposer node 仍可能参与选择 |
| BUG-13 | P0 | Apptainer `run()` API 与 ExecutionBackend 不兼容 | 开启 Apptainer 后参数报错 |
| BUG-14 | P0 | Apptainer command 使用 host path | `--containall` 下路径不可见 |
| BUG-15 | P0 | Solver/Self-edit/Validator 仍走 subprocess | 不是 end-to-end Apptainer |
| BUG-16 | P0 | `--network=none` 与在线 LLM API 冲突 | 模型调用失败 |
| BUG-17 | P0 | `--cleanenv` 下 LLM env 注入不可靠 | API key / endpoint 可能丢失 |
| BUG-18 | P1 | RepoChain 新 stage 文件多数仍是 stub | 目录已重构，逻辑仍集中旧 generator |
| BUG-19 | P1 | mutation backend weights 对真实 RepoChain 可能无效 | 配置与真实行为不一致 |
| BUG-20 | P1 | RepoChain-specific counters 没真正写入 summary | special detector 部分仍是死逻辑 |
| BUG-21 | P1 | stochasticity/localization metrics 没真实数据 | detector 名义存在但不触发 |
| BUG-22 | P1 | Evidence 读取 20k，实际只给 Diagnoser 500 chars | Joint Diagnosis 信息不足 |
| BUG-23 | P1 | success contrast 只有一句“solved” | 无法做 failure/success 对照 |
| BUG-24 | P1 | `diagnose_model` / Proposer model 没显式传递 | 实验配置不可追踪 |
| BUG-25 | P1 | Evolvable proposer transport schema 可能漂移 | Child 修改 schema 后 Controller 解析失败 |
| BUG-26 | P1 | Repo execution image ownership 未统一 | RepoChain contract test 可能缺依赖 |

---

# 2. BUG-01：修复 `TaskGenerationContext.level1_result`

当前 `level1_result=None` 没有类型标注，不会成为 dataclass field，但调用方会传 `level1_result=...`。

建议修改：

```python
from typing import Optional

@dataclass
class TaskGenerationContext:
    node: NodeRecord
    parent: Optional[NodeRecord] = None
    level1_result: Optional["Level1Result"] = None

    parent_failure_trajectories: List[str] = field(default_factory=list)
    current_child_level1_trajectories: List[str] = field(default_factory=list)

    parent_task_ids: List[str] = field(default_factory=list)
    parent_solved_task_ids: List[str] = field(default_factory=list)

    run_id: str = "run"
    output_dir: Optional[Path] = None
    model: str = "deepseek/deepseek-chat"
    task_store_dir: str = "./task_store"
    bootstrap: bool = False
```

这里同时为后续 5+5 quota 修复拆分 trajectory source。

验收测试：

```python
def test_task_generation_context_accepts_level1_result():
    ctx = TaskGenerationContext(node=node, level1_result=level1)
    assert ctx.level1_result is level1
```

---

# 3. BUG-02/03：真正把 RepoChainWorkflow 接入 runtime

当前实际调用链仍然是：

```text
proposer_main
    ↓
SWESmithEngine
    ↓
ProposerRunner
    ↓
ProposerPlanner
    ↓
lm_modify / procedural / pr_replay
    ↓
SWESmithEngine.generate()
```

虽然已有：

```text
proposer/workflows/repo_chain/RepoChainWorkflow
```

但 runtime 没真正经过它。

正确调用链应为：

```text
proposer_main
    ↓
build current node workflow
    ↓
RepoChainWorkflow
    ├── Weakness Analysis
    ├── Repository Transfer
    ├── Semantic Chain Discovery
    ├── Contract Generation
    ├── Chain Mutation
    │      ↓
    │  Mutation Backend
    ├── Causal Ablation
    ↓
Candidate
```

建议接口：

```python
class ProposerWorkflow(Protocol):
    def generate_batch(self, request: ProposerRequest) -> ProposerResult:
        ...

    def bootstrap(self, request: ProposerRequest) -> ProposerResult:
        ...
```

`ProposerRunner` 改为：

```python
class ProposerRunner:
    def __init__(self, workflow: ProposerWorkflow):
        self.workflow = workflow

    def generate_batch(self, request):
        if request.bootstrap:
            return self.workflow.bootstrap(request)
        return self.workflow.generate_batch(request)
```

`SWESmithEngine` 只保留为低层 mutation backend registry。

```python
mutation_engine = SWESmithEngine(agent_adapter=agent_adapter)

workflow = RepoChainWorkflow(
    agent_adapter=agent_adapter,
    engine=mutation_engine,
    trajectory_analyzer=TrajectoryAnalyzer(),
    code_locator=CodeLocator(),
)

runner = ProposerRunner(workflow=workflow)
```

## `initial_workflow` 的处理

不要让 Trusted Controller 永久硬编码 RepoChain。

建议 Node 内新增：

```text
initial_agent/src/proposer/workflow_registry.py
```

Root 默认：

```text
workflow = repo_chain
```

Child 可以通过 self-edit 修改 workflow registry、RepoChain implementation 或增加新 workflow。

Trusted Controller 只执行：

```text
python -m proposer.proposer_main
```

不直接 import RepoChain 内部实现。

验收：真实 proposer trace 必须出现：

```text
workflow=repo_chain
stage=weakness_analysis
stage=repository_transfer
stage=chain_discovery
stage=contract_generation
stage=chain_mutation
stage=causal_ablation
```

---

# 4. BUG-04/05：重新实现 Root Bootstrap

当前 `build_bootstrap_plans([], repo_spec)` 会直接产生 0 plans。

即使传 capability prior，现逻辑仍可能是：

```text
Capability Prior
    ↓
BugGenerationPlan(strategy="lm_modify")
    ↓
SWESmithEngine.generate()
```

这不是 RepoChain bootstrap。

正确流程：

```text
Bootstrap Capability Prior
    ↓
Repository / Subsystem Selection
    ↓
Semantic Chain Discovery
    ↓
Contract Generation
    ↓
Chain Mutation
    ↓
Causal Ablation
    ↓
Trusted Validation
    ↓
T_0
    ↓
S_0(T_0)
```

建议：

```python
class RepoChainWorkflow:
    def bootstrap(self, request: ProposerRequest) -> ProposerResult:
        ...
```

内部：

```python
for capability in BOOTSTRAP_CAPABILITY_PRIOR:
    anchors = self.bootstrap_target_selector.locate(
        capability=capability,
        repo_index=repo_index,
    )

    for anchor in anchors:
        seed_plan = build_bootstrap_seed_plan(
            capability=capability,
            anchor=anchor,
        )
        candidates += self.generate_from_seed_plan(seed_plan, ...)
```

Bootstrap plan 至少需要：

```text
target_repo_id
target_base_commit
anchor file / subsystem
capability_gap
required_topology
```

不能直接创建没有 target anchor 的 LM Modify plan。

K=10 时，不应限制“一 capability 只生成一题”。同一 capability 可以迁移到不同 subsystem / semantic chain。

验收：

```text
Root bootstrap=True
无外部 trajectory
生成 K trusted-valid tasks
Level2 恰好 K outcomes
Root COMPLETE
```

---

# 5. BUG-06：真正 enforce Causal Ablation

当前新 `CausalAblationStage.run()` 永远返回通过；旧 RepoChainGenerator 虽计算 causal ablation，但主要作为 metadata diagnostic。

当：

```yaml
require_causal_ablation: true
```

必须 reject：

```text
repair one required file
→ all target contracts pass
```

建议结果结构：

```python
@dataclass
class CausalAblationResult:
    passed: bool
    repair_one_file_results: dict[str, bool]
    all_single_file_repairs_still_fail: bool
    independently_active_file_count: int
    reason: str = ""
```

最低标准：

```text
2–6 modified production files
至少 2 个文件具有独立因果贡献
修复任意单个 required file 都不能完全恢复 target contract
```

推荐两层检查：

```text
RepoChain local ablation
    ↓
Candidate
    ↓
Trusted CandidateValidator authoritative ablation
```

---

# 6. BUG-07：修复 Solver empty-patch 统计

当前 `EvaluationOutcome.patch_path` 为空，而真实 patch 只写进 `trajectory_eval.json["model_patch"]`。`_solver_stats()` 再按 `patch_path` 读取，因此可能把所有任务算成 empty patch。

建议正式保存：

```text
model_patch.diff
```

并设置：

```python
patch_file = artifact_dir / "model_patch.diff"
patch_file.write_text(solver_patch, encoding="utf-8")
outcome.patch_path = str(patch_file)
```

`test_only_patch` 不要用字符串包含 `test` 判断，应解析 changed files：

```python
changed_files = extract_changed_files(patch)
production_files = [p for p in changed_files if not is_test_path(p)]
test_only = bool(changed_files) and not production_files
```

---

# 7. BUG-08/09：真正实现 5 Parent + 5 Current Child

当前 quota 只存在配置，没有真正控制生成；Parent 和 Child trajectories 也混在同一列表。

正确上下文：

```python
parent_failure_trajectories: List[str]
current_child_level1_trajectories: List[str]
```

Parent failure 只取：

```text
Parent Level2 unresolved trajectories
```

Current Child weakness 只取：

```text
Child Level1 unresolved / forgotten trajectories
```

TaskBatchBuilder 分两路：

```python
parent_tasks = generate_from_source(
    source_type="parent_failure",
    trajectories=context.parent_failure_trajectories,
    quota=5,
)

child_tasks = generate_from_source(
    source_type="current_child_level1",
    trajectories=context.current_child_level1_trajectories,
    quota=5,
)
```

不足时允许动态 fallback：

```text
5+5 → 4+6 → 3+7 → ...
```

但必须在 metadata 记录 fallback。

Candidate 必须携带：

```python
generation_metadata = {
    "source_node_id": ...,
    "source_trajectory_ids": [...],
    "source_type": ...,
}
```

Task commit 不得继续：

```text
source_trajectory=""
```

---

# 8. BUG-10/11：恢复 HGM-style Thompson Sampling

当前默认 `EpsilonGreedySelector` 必须从主实验移除。

## 8.1 NodeRecord 增加 utility measurements

```python
class NodeRecord(BaseModel):
    ...
    utility_measures: List[float] = []
    evaluated_task_ids: List[str] = []
    selection_eligible: bool = True
```

第一版 `utility_measures` 使用二元：

```text
1.0 = solved
0.0 = unresolved
```

建议来源仅为：

```text
Level2 trusted solver outcomes
```

Level1 继续作为 regression gate，不重复加入 TS posterior。

Root bootstrap Level2 outcomes 用于初始化 Root utility measures。

## 8.2 HGM-style descendant evaluations

新增：

```python
def pseudo_descendant_evals(node, num_pseudo):
    own = list(node.utility_measures)
    if len(own) < num_pseudo:
        return own
    mean = sum(own) / len(own)
    return [mean] * num_pseudo
```

```python
def descendant_evals(node, archive, num_pseudo):
    evals = pseudo_descendant_evals(node, num_pseudo)
    for desc in archive.descendants_of(node.node_id):
        evals.extend(desc.utility_measures)
    return evals
```

Selector：

```python
class ThompsonSamplingSelector:
    def __init__(self, num_pseudo_descendant_evals: int = 10):
        self.num_pseudo = num_pseudo_descendant_evals

    def select(self, candidates, archive, rng):
        best_node = None
        best_theta = -1.0

        for node in candidates:
            evals = descendant_evals(node, archive, self.num_pseudo)

            if not evals:
                alpha = 1.0
                beta = 1.0
            else:
                successes = sum(evals)
                failures = len(evals) - successes
                alpha = 1.0 + successes
                beta = 1.0 + failures

            theta = rng.betavariate(alpha, beta)

            if theta > best_theta:
                best_theta = theta
                best_node = node

        return best_node
```

## 8.3 修 ParentSelector API

当前 Orchestrator 先按 scoring mode 过滤一次，但 selector 内部又重新调用 `eligible_parents()`，可能绕过 quality gate。

改为：

```python
class ParentSelector(Protocol):
    def select(
        self,
        candidates: list[NodeRecord],
        archive: NodeArchive,
        rng: random.Random,
    ) -> NodeRecord:
        ...
```

Orchestrator：

```python
candidates = self.archive.eligible_parents(
    min_solved=self.config.scoring.min_parent_solved_tasks,
    scoring_mode=self.config.scoring.mode,
)

return self.selector.select(
    candidates=candidates,
    archive=self.archive,
    rng=self.rng,
)
```

## 8.4 配置

```yaml
selection:
  strategy: thompson_sampling
  num_pseudo_descendant_evals: 10
```

主实验固定：

```text
thompson_sampling
```

`epsilon_greedy` 仅作为 ablation。

## 8.5 动态任务的可比性

Gödel0 不同节点面对不同 Proposer tasks，因此 Level2 binary outcomes 存在 difficulty bias。

第一版建议：只有满足以下 gate 的 Node，其 Level2 outcomes 才进入 TS：

```text
trusted-valid task batch
causal ablation pass
valid yield pass
difficulty calibration pass
```

更严格的后续版本可维护 Global Validated Task Pool，让节点在共享任务池上积累 TS measurements；但第一版先采用 quality-gated Level2 outcomes。

---

# 9. BUG-12：真正接通 HGM-style quality gate

当前 `compute_scores()` 返回 `eligible`，但 Orchestrator 没使用，且没有传 `valid_yield`、`causal_ablation_pass`、`batch_complete`。

修改：

```python
scores = compute_scores(
    retention_rate=r,
    frontier_accuracy=p,
    ...,
    valid_yield=valid_yield,
    causal_ablation_pass=causal_ablation_pass_rate,
    batch_complete=batch_result.complete,
)
```

NodeRecord 保存：

```python
selection_eligible = scores.eligible
```

Archive：

```python
if scoring_mode == "hgm" and not node.selection_eligible:
    continue
```

不要继续用 `proposer_score > 0` 代替完整 gate。

主实验 parent selection：

```text
Quality Gate
    ↓
Eligible Nodes
    ↓
Thompson Sampling
```

`node_score` 仅保留用于分析或 ablation。

---

# 10. BUG-13~17：Apptainer 全链路修复

## 10.1 统一 ExecutionBackend API

当前 `ExecutionBackend.run()` 没有 `image` 参数，而 `ApptainerRunner.run()` 要求必填 `image`，破坏 polymorphism。

建议把 image 放 constructor：

```python
class ApptainerRunner(ExecutionBackend):
    def __init__(
        self,
        image: Path,
        apptainer_bin: str = "apptainer",
        clean_env: bool = True,
        network_disabled: bool = False,
    ):
        self.image = Path(image)
```

`run()` 与 SubprocessRunner 完全同签名。

## 10.2 容器内必须使用 container path

不要 bind 后仍把 host absolute path 传进 command。

容器分支应构造：

```text
python /agent/coding_agent.py
--git_dir /workspace
--outdir /outputs
--chat_history_file /logs/trajectory.jsonl
```

Apptainer 使用：

```text
--pwd /workspace
```

## 10.3 标准 mount layout

```text
agent code      → /agent
task repo       → /workspace
outputs         → /outputs
logs            → /logs
control/request → /control
```

所有容器 command 只引用这些 container paths。

## 10.4 网络策略分 phase

Agent-facing phase：

```text
Solver
Proposer
Diagnoser
Self-Improve
```

需要在线 LLM API 时：

```text
network_enabled = true
```

Trusted repository tests：

```text
network_disabled = true
```

不要统一 `--network=none`。

## 10.5 `--cleanenv` 下显式传 env

不要只 `full_env.update(env)`。

用 allowlist 显式：

```text
--env OPENAI_API_KEY=...
--env DEEPSEEK_API_KEY=...
--env DEEPSEEK_API_BASE_URL=...
--env VLLM_HOST=...
```

或封装 `APPTAINERENV_*`。

## 10.6 区分 Agent Image 与 Repo Image

推荐：

```text
Self-Improve / Diagnosis:
  Agent Image

Solver on task:
  Repo-specific Image + bind Node agent code

RepoChain generation:
  Repo-specific Image

Trusted Validator:
  Repo-specific Image
```

第一版只有 Ansible 时，可直接构建一个固定 Ansible `.sif`，包含 repo runtime dependencies 与 LLM client dependencies。

## 10.7 不允许 nested Apptainer

不要 Proposer 容器内再 `apptainer exec`。

RepoChain 内部 `_run_command()` 可以继续 subprocess，前提是整个 Proposer process 已经在正确 Repo image 中。

## 10.8 CandidateValidator 必须接 ExecutionBackend

所有：

```text
clean test
bugged test
reverse test
F2P/P2P
causal ablation
```

统一通过 repo backend。

## 10.9 CommonAgentAdapter 必须接 backend factory

当前 `CommonAgentAdapter()` 默认自行创建 `SubprocessRunner`。

应创建：

```python
agent_backend = backend_factory.agent_backend()
repo_backend = backend_factory.repo_backend(repo_id)
```

并显式注入。

## 10.10 推荐 Backend Factory

```python
class ExecutionBackendFactory:
    def agent_backend(self): ...
    def repo_backend(self, repo_id: str): ...
```

这样 Solver、Proposer、Validator、Self-edit 都由同一个配置源构建 backend。

---

# 11. BUG-18/19：RepoChain stage stub 与 mutation backend weights

当前 stage package 已经建立，但大量真实逻辑仍在 `swesmith/repo_chain.py`。

第一阶段不要求立刻完全搬迁。

先保证：

```text
ProposerRunner
→ RepoChainWorkflow
→ backing RepoChainGenerator
```

运行语义正确后，再逐步拆 stage。

当前 `mutation_backends` 权重可能并不真正影响 backing RepoChainGenerator。

正式 v1 建议二选一：

### 短期推荐

RepoChain 固定使用当前 chain mutation，暂时把 mutation backend weights 标记为 experimental/unused。

### 后续

让 Stage 5 使用 chain-aware backend：

```python
generate_chain(
    chain_plan,
    mutation_sites,
    contracts,
    ...
)
```

普通单文件 backend 不能直接替代 multi-file RepoChain mutation。

---

# 12. BUG-20：RepoChain-specific stats 必须持久化

`TaskBatchResult` 增加：

```python
repo_chain_stats: dict = field(default_factory=dict)
```

建议：

```json
{
  "contract_generation_failure_count": 3,
  "clean_contract_failure_count": 2,
  "mutation_materialization_failure_count": 4,
  "causal_ablation_failure_count": 1,
  "no_f2p_count": 6,
  "no_p2p_count": 0,
  "duplicate_count": 1,
  "statement_leakage_count": 0
}
```

生成端 structured emit，`generation_summary.json` 原样保存。

SpecialDetector 直接读 structured counters，不要主要依赖 rejection string substring。

---

# 13. BUG-21：stochasticity / localization detector

当：

```text
solver_rollouts < 2
```

建议关闭 stochasticity alert。

后续基于：

```text
same task + same node + multiple rollouts/seeds
```

统计稳定性。

Localization collapse 应从 trajectory 结构化判断，例如：

```text
no production file opened
no relevant symbol located
patch unrelated
repeated search without narrowing
```

不要保留永远为 0 的 counter。

---

# 14. BUG-22/23：修复 Joint Diagnosis Evidence

当前读取 20k excerpt 后，最终 `EvidenceItem.summary` 仍只保存前 500 chars。

建议 `EvidenceItem` 增加：

```python
raw_text: Optional[str]
```

或：

```python
raw_excerpt_path: Optional[str]
```

Diagnoser 在总 budget 内读取真正 representative raw evidence。

推荐单项大小：

```text
primary failure: 4k–8k chars
supporting failure: 2k–4k
success contrast: 2k–4k
candidate rejection: 2k–4k
```

Success contrast 不能只是：

```text
SOLVED task X
```

应包含：

```text
tool sequence
files inspected
patch files
test behavior
trajectory excerpt
```

形成真正 failure/success 对照。

---

# 15. BUG-24：模型配置必须显式

建议：

```yaml
models:
  solver_model: "..."
  proposer_model: "..."
  diagnose_model: "..."
  self_improve_model: "..."
```

即使都相同也显式记录。

所有 `chat()` 调用必须显式传 model，不允许内部 fallback 到另一个默认模型。

每次 artifact 保存：

```text
model
temperature
max_tokens
seed
```

---

# 16. BUG-25：稳定 Proposer Transport Contract

当前 Controller 使用 root 侧 `ProposerResult` schema，但 Child 可能 self-edit `proposer/request.py`，存在协议漂移风险。

建议把 trust-boundary transport 移到：

```text
src/godel0/schemas/proposer_transport.py
```

固定：

```text
ProposerRequestV1
ProposerResultV1
CandidateTransportV1
```

Node workflow 可以自由 evolve，但 transport schema 不允许 self-edit。

另一种方案是把 `proposer/request.py` 加入 PatchGuard protected paths，但推荐 trusted schema 方案。

---

# 17. BUG-26：Repo Image 与 Ansible 环境

RepoProfile 只负责语义：

```text
source roots
test roots
contract style
entrypoints
```

RepoSpec / RepoEnvironment 负责：

```text
repo path
base commit
image path
trusted test command
timeout
```

建议 RepoSpec 增加：

```python
image: Optional[str]
```

例如：

```yaml
repo_id: ansible
image: /images/ansible_<commit>.sif
```

所有 Ansible contract/test 均使用固定 image。

---

# 18. Thompson Sampling 与 Scoring 的最终关系

推荐主方法：

```text
Level1 retention
    ↓
Regression Gate

RepoChain task quality
    ↓
Proposer Quality Gate

Level2 outcomes
    ↓
utility_measures

Eligible Nodes
    ↓
HGM-style Thompson Sampling
    ↓
Parent for expansion
```

`a × b` 可以保留为分析指标或 ablation，但正式主实验 Parent Selection 应由 Thompson Sampling posterior 驱动。

---

# 19. 推荐修复后的主循环

```python
def run():
    ensure_root_bootstrap()

    while not budget.exhausted():
        eligible = archive.eligible_parents(
            min_solved=config.scoring.min_parent_solved_tasks,
            require_quality_gate=True,
        )

        parent = thompson_selector.select(
            candidates=eligible,
            archive=archive,
            rng=rng,
        )

        diagnosis = prepare_joint_diagnosis(parent)
        child = self_edit(parent, diagnosis)

        if not child_build_gate(child):
            continue

        level1 = evaluate_regression(parent, child)

        if not level1.passed:
            record_failed_child_evidence(child)
            continue

        task_batch = proposer_task_provider.get_tasks(
            child,
            build_task_context(
                parent_failure_trajectories=...,
                current_child_level1_trajectories=...,
            ),
        )

        if not task_batch.complete:
            record_failed_child_evidence(child)
            continue

        level2 = evaluate_level2(child, task_batch)
        quality = compute_proposer_quality(task_batch)

        child.selection_eligible = quality.passed
        child.utility_measures = [
            1.0 if outcome.resolved else 0.0
            for outcome in level2.outcomes
        ]

        child.status = COMPLETE
        archive.update(child)
```

Root：

```python
def ensure_root_bootstrap():
    T0 = root_proposer.repo_chain_bootstrap()
    L2_root = evaluate(root, T0)

    root.utility_measures = [
        int(outcome.resolved)
        for outcome in L2_root.outcomes
    ]

    root.selection_eligible = quality_gate(T0)
    root.status = COMPLETE
```

---

# 20. 推荐修改顺序

## Phase A：立即阻断 bug

```text
A1. TaskGenerationContext dataclass
A2. Root bootstrap empty prior
A3. RepoChainWorkflow runtime wiring
A4. Root bootstrap must call workflow.bootstrap
A5. Solver patch_path / empty patch stats
```

完成后先跑 Root Bootstrap E2E。

## Phase B：恢复核心算法

```text
B1. ThompsonSamplingSelector
B2. Node utility_measures
B3. descendant eval aggregation
B4. quality gate
B5. selector candidate filtering
```

## Phase C：RepoChain task quality

```text
C1. enforce causal ablation
C2. structured repo_chain_stats
C3. 5+5 quota
C4. trajectory provenance
```

## Phase D：Diagnosis

```text
D1. special detector real metrics
D2. evidence raw excerpts
D3. real success contrast
D4. explicit model config
```

## Phase E：Apptainer

```text
E1. unify ExecutionBackend API
E2. backend factory
E3. container path mapping
E4. image registry
E5. network policy
E6. env allowlist
E7. Solver Apptainer
E8. Proposer Apptainer
E9. Validator Apptainer
E10. Self-edit Apptainer
```

---

# 21. 必须新增的测试

## Unit Tests

```text
test_task_generation_context_fields

test_thompson_sampling_beta_posterior

test_thompson_sampling_descendant_aggregation

test_thompson_sampling_uses_passed_candidates

test_root_bootstrap_uses_capability_prior

test_root_bootstrap_enters_repochain

test_repochain_workflow_called

test_causal_ablation_rejection

test_solver_empty_patch_stats

test_test_only_patch_stats

test_parent_child_quota

test_task_source_provenance

test_hgm_quality_gate_used

test_repochain_stats_persisted

test_model_config_forwarding
```

## Apptainer Integration Tests

必须在真实 cluster 节点运行：

```text
test_apptainer_basic_exec

test_apptainer_bind_paths

test_apptainer_clean_env_allowlist

test_apptainer_llm_network

test_apptainer_solver_one_task

test_apptainer_proposer_one_candidate

test_apptainer_repochain_clean_contract

test_apptainer_candidate_validator

test_apptainer_self_edit_smoke
```

## Full E2E

正式实验前必须通过：

```text
Root
    ↓
RepoChain bootstrap
    ↓
K trusted-valid tasks
    ↓
Root Level2
    ↓
Root utility measurements
    ↓
Thompson Sampling selects Root
    ↓
Joint Diagnosis
    ↓
Self-edit
    ↓
Child
    ↓
Level1 pass
    ↓
RepoChain generates K tasks
    ↓
Level2
    ↓
Child COMPLETE
    ↓
Child appears in TS candidate set
```

至少完成 `Root + 1 completed child`，再开始 20 epoch。

---

# 22. 建议增加的运行时 assertions

```python
assert isinstance(task_provider, ProposerTaskProvider)
assert proposer_trace.workflow == "repo_chain"
assert len(level2.outcomes) == batch_size
assert len(child.utility_measures) == batch_size
assert all(v in (0, 1, 0.0, 1.0) for v in child.utility_measures)
assert task_batch.complete
assert all(task.source_type for task in task_batch.tasks)
assert all(
    task.source_trajectory or task.source_type == "bootstrap"
    for task in task_batch.tasks
)

if config.proposer.repo_chain.require_causal_ablation:
    assert task_batch.causal_ablation_pass_rate >= threshold

if config.execution.backend == "apptainer":
    assert backend_factory.is_apptainer
```

这些 assertions 可以避免“程序能跑，但实际跑的不是设计中的 Gödel0”。

---

# 23. 正式实验前 Preflight Checklist

```text
[ ] Root bootstrap 不使用外部 trajectory
[ ] Root bootstrap 确实经过 RepoChainWorkflow
[ ] RepoChainWorkflow runtime trace 可见
[ ] Root 产生 K 个 trusted-valid tasks
[ ] Root Solver 有 K 个完整 outcomes
[ ] utility_measures 正确写入 Node
[ ] ParentSelector 是 ThompsonSamplingSelector
[ ] descendant evaluations 正确
[ ] epsilon-greedy 未用于主实验
[ ] Level1 regression gate 生效
[ ] Proposer quality gate 生效
[ ] causal ablation 真正 enforce
[ ] 5+5 source quota 真正 enforce
[ ] source trajectory provenance 完整
[ ] special detectors 有真实数据
[ ] evidence 包含真实 raw trajectory
[ ] model 配置显式记录
[ ] Apptainer image 明确
[ ] container path 不使用 host absolute path
[ ] LLM phase network 可访问
[ ] trusted test phase network disabled
[ ] API env 通过 allowlist 注入
[ ] Solver 使用 Apptainer
[ ] Proposer 使用 Apptainer
[ ] CandidateValidator 使用 Apptainer
[ ] Self-edit 使用 Apptainer
[ ] Root + 1 Child E2E 完整通过
```

---

# 24. 最终推荐的 Gödel0 定义

修完以后，整个系统应可以准确描述为：

> **Gödel0 保留 HGM-style agent testing、joint diagnosis、self-modification 与 Thompson Sampling tree search，但将固定 benchmark task pool 替换为由当前节点中的 Evolvable Proposer 动态产生的 Coding Tasks。Proposer 默认采用 RepoChain workflow，从 Solver failure 中抽取能力缺陷，在大型真实 repository 中寻找新的 semantic chain，通过 contract generation、multi-file mutation 与 causal ablation 构造高难度任务。所有任务通过 trusted validation 后，以与 benchmark task 相同的接口交给 Solver。Node 的可信 Level2 outcomes 形成 Thompson Sampling 的 utility measurements，从而继续驱动 HGM-style tree evolution。**
