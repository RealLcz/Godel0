# Gödel0 架构收束与重构规划：HGM-style Self-Improvement + Evolvable RepoChain Proposer

## 1. 最终研究定位

Gödel0 的主体应保持 HGM-style self-improvement 逻辑：

```text
Coding Tasks
    ↓
Agent Evaluation
    ↓
Failure / Cycle Diagnosis
    ↓
Self-Improvement Coding Task
    ↓
Coding Agent 修改自身
    ↓
重新 Evaluation
    ↓
进入 Archive / Tree Search
```

与 HGM 的主要区别不是改变 self-improvement 主循环，而是把固定 Benchmark Task Provider 替换为一个可进化的 Proposer：

```text
HGM:
Static Benchmark
    ↓
Agent Test
    ↓
Diagnose
    ↓
Self-Modify
    ↓
Agent Test

Gödel0:
Evolvable Proposer
    ↓
RepoChain-generated Coding Tasks
    ↓
Agent Test
    ↓
Joint Diagnosis
    ↓
Self-Modify
    ↓
Evolved Proposer generates new Coding Tasks
    ↓
Agent Test
```

因此，Gödel0 可以定义为：

> **Gödel0 是一个 HGM-style self-improving coding-agent system，其中固定 benchmark evaluator 被一个与 Agent 共同版本化、可通过 self-modification 改进的 Proposer 所替代。Proposer 默认遵循 RepoChain workflow，将 Solver 暴露出的抽象能力缺陷迁移到新的 repository-level semantic chains，生成高难度 Coding Tasks。生成任务经过 trusted validation 后，以与标准 benchmark task 相同的接口交给 Solver。Solver、Proposer 和共享工具的闭环日志共同驱动下一轮 self-improvement。**

---

# 2. 最终总体架构

```text
                         Parent Node N_i
                               │
                               │
                    读取历史测试与生成结果
                               │
                               ▼
                       Joint Cycle Diagnosis
                    找到一个最高价值 root cause
                               │
                               ▼
                    HGM-style Self-Improvement
                    Coding Agent 修改自身代码
                               │
                               ▼
                         Child Node N_j
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     │
             Level 1 Regression           │
          Child 做 Parent 已解决任务        │
                    │                     │
                pass│                     │
                    ▼                     │
              Child Proposer              │
           使用 RepoChain Workflow         │
                    │                     │
                    ▼                     │
        生成新的 Coding Tasks T_j          │
                    │                     │
                    ▼                     │
             Trusted Validation           │
                    │                     │
                    ▼                     │
          Child Solver 解决 T_j            │
                    │                     │
                    ▼                     │
                 Level 2                  │
                    │                     │
                    ▼                     │
              Node Utility                │
                    │                     │
                    └────────→ Archive ───┘
```

节点仍然是一个联合节点：

```text
Node N_i
=
{
    Solver implementation,
    Proposer implementation,
    RepoChain workflow implementation,
    mutation backends,
    shared tools,
    prompts,
    runtime behavior
}
```

只有一棵 Node Tree。

不建立：

```text
Solver Tree
+
Proposer Tree
```

也不在 self-improvement 前执行：

```text
choose_solver_or_proposer()
```

---

# 3. RepoChain 的正确定位

当前实现中，`repo_chain` 与下面这些策略处于同一级：

```text
lm_modify
lm_rewrite
procedural
combine
pr_mirror
pr_replay
repo_agent
repo_chain
```

这不符合最终设计。

RepoChain 不应该是一个普通 bug generation strategy。

正确关系应为：

```text
RepoChain
=
Proposer 的默认 Task Generation Workflow

LM Modify / Procedural / PR Replay
=
RepoChain 内部使用的 Mutation Backend
```

即：

```text
Proposer
    │
    ▼
RepoChainWorkflow
    │
    ├── Weakness Identification
    ├── Repository Transfer
    ├── Semantic Chain Discovery
    ├── Contract Generation
    ├── Mutation Planning
    │       │
    │       ├── LM-based Mutation
    │       ├── Procedural Mutation
    │       └── PR Replay
    │
    ├── Causal Ablation
    └── Task Candidate
             │
             ▼
       Trusted Validator
```

RepoChain 是一种**工作思想、推理范式和任务生成 workflow**，不是 Agent Tool。

---

# 4. 第一项重构：把 RepoChain 提升为 Proposer Workflow

建议目录调整为：

```text
initial_agent/src/
├── coding_agent.py
├── llm_withtools.py
├── llm.py
├── tools/
├── prompts/
│
├── proposer/
│   ├── proposer_main.py
│   ├── runner.py
│   ├── schemas.py
│   ├── workflow.py
│   │
│   ├── workflows/
│   │   └── repo_chain/
│   │       ├── workflow.py
│   │       ├── weakness_analysis.py
│   │       ├── capability_targeting.py
│   │       ├── repository_transfer.py
│   │       ├── chain_discovery.py
│   │       ├── contract_generation.py
│   │       ├── mutation_planning.py
│   │       ├── causal_ablation.py
│   │       └── prompts/
│   │
│   └── repo_profiles/
│       ├── base.py
│       └── ansible.py
│
└── swesmith/
    ├── mutations/
    │   ├── procedural.py
    │   ├── lm_modify.py
    │   ├── lm_rewrite.py
    │   ├── pr_replay.py
    │   └── combine.py
    └── patch_utils.py
```

这里：

```text
proposer/workflows/repo_chain/
```

保存 RepoChain 的核心工作逻辑。

```text
swesmith/mutations/
```

只保存底层 mutation mechanism。

---

# 5. 配置系统也要区分 Workflow 和 Mutation Backend

不要继续写成：

```yaml
proposer:
  strategies:
    repo_chain: 1.0
```

建议改为：

```yaml
proposer:
  initial_workflow: repo_chain

  repo_chain:
    min_files: 2
    max_files: 6

    min_mutation_sites: 3
    max_mutation_sites: 8

    context_file_budget: 10

    require_generated_contracts: true
    require_causal_ablation: true

    mutation_backends:
      lm_modify: 0.5
      procedural: 0.2
      pr_replay: 0.3
```

语义变成：

```text
workflow:
  RepoChain

mutation backend:
  LM / Procedural / PR Replay
```

未来 Proposer self-evolution 可以：

```text
调整 RepoChain prompt；
修改 chain discovery；
改变 context selection；
调整 mutation backend；
新增新的 backend；
修改 contract generation；
甚至创造新的 workflow。
```

---

# 6. RepoChain 必须属于可进化 Node，而不是 Trusted Controller

Trusted Controller 不应该直接知道 RepoChain 内部逻辑。

错误设计：

```python
controller = RepoChainWorkflow(...)
controller.generate_tasks(...)
```

正确设计：

```python
task_provider.get_tasks(node, context)
```

然后：

```text
Trusted Controller
        │
        ▼
ProposerTaskProvider
        │
        ▼
当前 Node commit 中的 proposer_main.py
        │
        ▼
load_current_workflow()
        │
        ▼
RepoChainWorkflow
```

Root Node 默认：

```text
workflow = RepoChain
```

Child Node 可以通过 self-improvement 修改：

```text
proposer/
swesmith/
tools/
prompts/
coding_agent.py
```

因此：

```text
N_i
    ↓ self-improvement
N_j
```

可能同时改进：

```text
Solver reasoning
Proposer reasoning
RepoChain workflow
Mutation backend
Shared tools
Context management
```

但仍然只有一次统一的 Node self-modification。

---

# 7. 第二项重构：建立 HGM-compatible TaskProvider 接口

整个 Agent Test 接口应与 HGM / SWE-bench 保持一致。

定义：

```python
class TaskProvider(Protocol):
    def get_tasks(
        self,
        node: NodeRecord,
        context: TaskGenerationContext,
    ) -> TaskBatch:
        ...
```

HGM：

```python
BenchmarkTaskProvider
```

返回固定 Benchmark Tasks。

Gödel0：

```python
ProposerTaskProvider
```

调用当前 Node 的 Proposer，返回 RepoChain-generated Coding Tasks。

Controller 的后续逻辑完全一致：

```python
tasks = task_provider.get_tasks(node, context)

results = evaluator.evaluate(
    agent=node,
    tasks=tasks,
)

diagnosis = diagnose(
    node=node,
    results=results,
)

child = self_improve(
    parent=node,
    diagnosis=diagnosis,
)
```

这样可以清楚表达：

```text
HGM:
TaskProvider = Static Benchmark

Gödel0:
TaskProvider = Evolvable Proposer
```

---

# 8. Coding Task 接口必须保持统一

无论任务来自：

```text
SWE-bench
```

还是：

```text
RepoChain
```

对 Solver 都应该表现为：

```python
TaskInstance(
    repo_id,
    base_commit,
    problem_statement,
    test_description,
)
```

Solver 始终通过：

```text
coding_agent.py
    --problem_statement
    --git_dir
    --base_commit
    --test_description
```

执行任务。

Solver 不能知道：

```text
这个题是 benchmark 题；
还是 RepoChain 生成题；
bug patch 是什么；
hidden contracts 是什么；
mutation sites 在哪里。
```

Trusted Controller 负责将任务 materialize 成标准 bugged repository。

这部分应该尽量复用 HGM / SWE-bench-style Agent evaluation interface。

---

# 9. 第三项修改：重新设计 Root Bootstrap

当前 Root 依赖：

```text
GODEL0_BOOTSTRAP_SOLVER_TRAJECTORY
```

不应该成为正式方法的一部分。

既然 Gödel0 的演化任务都由 Proposer 生成，Root 应支持：

```text
Bootstrap Mode
```

流程：

```text
P_0
    ↓
Bootstrap Capability Prior
    ↓
RepoChain
    ↓
生成 T_0
    ↓
S_0(T_0)
    ↓
获得第一批 Solver trajectories
    ↓
后续进入 trajectory-conditioned generation
```

Bootstrap capability prior 可以定义为通用 Coding Agent 能力：

```text
cross-file localization
multi-module state propagation
configuration precedence
error handling
compatibility preservation
API contract reasoning
multi-step repository reasoning
```

这些只是能力类别，不是固定 benchmark tasks。

从 N1 开始：

```text
Parent Solver failures
+
Current Child Level 1 behavior
+
历史 generation feedback
    ↓
Weakness Analysis
    ↓
RepoChain
    ↓
T_i
```

---

# 10. 第四项修改：升级 Trajectory / Weakness Analysis

当前简单 heuristics：

```text
empty patch
no localization
no F2P
no tool calls
```

只能提供非常粗的 failure stage。

RepoChain 要生成 SWE-bench Verified / Pro 级别任务，建议输出更丰富的 Failure Signature：

```yaml
failure_stage: cross_file_reasoning

root_cause:
  Solver identified the consumer but failed to trace
  an identity-preserving state transition through
  intermediate modules.

capability_gap:
  multi_module_state_tracking

reasoning_pattern:
  localized_reasoning_without_dependency_tracing

code_topology:
  producer -> carrier -> transformer -> consumer

tool_behavior:
  repeated_local_search_without_call_chain_expansion

failed_fix_pattern:
  localized_patch_only

transfer_constraints:
  avoid_same_domain: true
  require_3_plus_modules: true
  require_behavioral_contract: true
```

建议 Failure Signature 至少包含：

```text
failure_stage
root_cause
capability_gap
reasoning_pattern
code_topology
tool_behavior
failed_fix_pattern
transfer_constraints
forbidden_copy_features
```

RepoChain 迁移的是：

```text
抽象能力缺陷
```

而不是：

```text
原任务的 repository subsystem
原 bug story
原 symbols
原 domain nouns
```

---

# 11. RepoChain 的正式八阶段 Workflow

## Stage 1：Weakness Identification

输入：

```text
Solver trajectories
Solver outcomes
Previous task metadata
```

输出：

```text
CapabilityGap
FailureSignature
```

## Stage 2：Repository Transfer

在 base repository 中寻找：

```text
需要相同抽象能力
+
属于不同代码子系统
```

的目标。

例如：

```text
原失败：
config propagation

新任务：
parser
→ intermediate state
→ executor
```

迁移的是 reasoning requirement，而不是业务故事。

## Stage 3：Semantic Chain Discovery

发现一个完整行为链：

```text
entrypoint
    ↓
producer
    ↓
carrier
    ↓
transformer
    ↓
consumer
    ↓
observable behavior
```

RepoChain 的真正核心是：

> 找到一个跨多个 production modules 的统一 semantic invariant。

## Stage 4：Contract Generation

先生成行为 Contract：

```text
Target Contract
+
Compatibility Control
```

并验证：

```text
Clean Repository
→ Contract Pass
```

建议至少包含：

```text
target behavior
compatibility behavior
observable output
```

不能使用 source-code inspection 代替行为验证。

## Stage 5：Chain Mutation

围绕一个统一：

```text
root invariant
```

在语义链上产生多个 mutation manifestations。

例如：

```text
2–6 production files
3–8 mutation sites
```

但必须满足：

```text
所有 mutation site
服务于同一个 behavioral regression
```

不能只是随机组合多个独立 bug。

Mutation Backend 可以包括：

```text
LM-based mutation
Procedural mutation
PR Replay
```

## Stage 6：Causal Ablation

必须验证任务确实具有 Chain-level 因果结构。

例如：

```text
repair only one file
→ complete contracts 仍然失败
```

以及：

```text
single isolated mutation
→ 是否独立触发 contract
```

目标是排除：

```text
多个无关单文件 bug
简单拼接成 multi-file task
```

建议将：

```text
causal_ablation_pass
```

作为 RepoChain Task 的核心质量信号。

## Stage 7：Trusted Validation

全部位于 Node 外：

```text
Clean pass
Bugged fail
F2P
P2P
Reverse restore
Duplicate check
Safety check
Leakage check
```

Agent 生成的本地测试结果不能作为最终 trusted result。

## Stage 8：Task Packaging

最终转换为统一 Coding Task：

```text
repo_id
base_commit
bug_patch
problem_statement
test_description
hidden contracts / F2P metadata
```

Solver 看到的接口与标准 benchmark task 一致。

---

# 12. 第五项修改：真正实现 HGM-style Special Failure System

Gödel0 应保留 HGM 的思想：

> 优先处理跨任务、系统性的失败模式，而不是总从普通 unresolved task 中诊断。

但不预先决定：

```text
evolve Solver
或
evolve Proposer
```

而是自动检测 Special Alerts，再做 Joint Diagnosis。

## 12.1 Solver-side Special Cases

至少实现：

```text
solver_empty_patch
solver_test_only_patch
solver_stochasticity
solver_context_overflow
solver_timeout
solver_repeated_tool_loop
solver_regression
solver_localization_collapse
```

## 12.2 Proposer / RepoChain-side Special Cases

建议实现：

```text
proposer_empty_task_batch
contract_generation_failure
clean_contract_failure
mutation_materialization_failure
no_f2p_dominant
no_p2p
causal_ablation_failure
low_valid_yield
duplicate_collapse
difficulty_too_easy
difficulty_too_hard
context_overflow
repo_subsystem_collapse
statement_leakage
```

特别重要：

```text
causal_ablation_failure
```

如果大量任务满足：

```text
修复一个文件
→ 全部 hidden contracts 恢复
```

说明 RepoChain 生成的任务没有真正达到 chain-level reasoning 难度。

---

# 13. Special Detector 必须真正接通数据

当前 Detector 接口虽然存在，但 Orchestrator 没有传足够数据。

应该改成：

```python
alerts = special_detector.detect(
    summary=summary,

    trajectories=cycle_artifacts.solver_trajectories,

    candidates=cycle_artifacts.candidate_reports,

    tool_events=cycle_artifacts.tool_events,

    solver_stats={
        "empty_patch_count": ...,
        "test_only_patch_count": ...,
        "evaluated_count": ...,
        "timeout_count": ...,
        "context_overflow_count": ...,
        "stochastic_task_count": ...,
    },

    proposer_stats={
        "empty_batch_count": ...,
        "contract_failure_count": ...,
        "clean_contract_failure_count": ...,
        "no_f2p_count": ...,
        "no_p2p_count": ...,
        "causal_ablation_failure_count": ...,
        "duplicate_count": ...,
        "statement_leakage_count": ...,
    },
)
```

Detector 只负责：

```text
检测异常
+
提供 severity / confidence
```

不直接决定修改哪个组件。

---

# 14. 第六项修改：升级 Joint Diagnosis Evidence

不要继续使用：

```text
前 4 个 Solver logs
+
前 4 个 Proposer candidates
```

应改为：

```text
Alert-conditioned Evidence Retrieval
```

例如，当主要异常是 `no_f2p_dominant`：

```text
Node Cycle Summary

Special Alert:
31 / 40 candidates rejected by no_f2p

Representative Proposer Evidence:
2 个 no_f2p candidates
1 个成功 F2P candidate
1 个对应 chain plan

Solver Evidence:
1 个成功 task trajectory
2 个失败 task trajectories
```

当主要异常是 `solver_empty_patch`：

```text
3 个 empty-patch Solver trajectories
1 个成功 patch trajectory
Task quality summary
RepoChain generation summary
```

当主要异常是 `causal_ablation_failure`：

```text
失败的 chain plan
mutation sites
repair-one-file ablation results
成功的 chain-level task 作为 contrast
```

最终：

```text
Special Alerts
    ↓
Evidence Retrieval
    ↓
Cycle Diagnoser
    ↓
ONE primary root cause
```

---

# 15. Self-Improvement 仍然保持统一 Joint Diagnosis

不增加：

```python
choose_solver_or_proposer()
```

正确流程：

```text
Node Cycle
    ↓
Auto Special Detection
    ↓
Representative Evidence
    ↓
Joint Diagnoser
    ↓
ONE highest-impact root cause
    ↓
生成一个 Coding Task
    ↓
同一个 Self-Improvement Coding Agent
修改完整 Node codebase
```

Diagnoser 可以最终建议修改：

```text
coding_agent.py
proposer/
repo_chain workflow
mutation backend
tools/
prompts/
runtime
```

但：

```text
一次 evolution
只处理一个 primary root cause
```

允许跨多个文件或模块修改，但所有修改必须服务同一根因。

---

# 16. 第七项修改：明确 Parent / Child Weakness Task Quota

建议每批 K=10：

```text
5 个主要来源于 Parent unresolved weaknesses
+
5 个主要来源于 Current Child Level 1 behavior
```

配置：

```yaml
task_generation:
  batch_size: 10

  sources:
    parent_failure:
      quota: 5

    current_child_level1:
      quota: 5
```

如果某一侧不足：

```text
5 + 5
↓
3 + 7
↓
0 + 10
```

允许动态 fallback。

但必须在 Task Metadata 中保存：

```text
source_node
source_trajectory
source_type
```

不能把所有 trajectory 混在一起后丢失来源信息。

---

# 17. 第八项修改：把 Ansible-specific 逻辑移到 RepoProfile

不要在通用 Planner 中写：

```python
if repo_id == "ansible":
    ...
```

建议：

```text
RepoChainWorkflow
      ↓
RepoProfileRegistry
      ↓
AnsibleProfile
```

接口：

```python
class RepoProfile:
    def source_roots(self): ...
    def test_roots(self): ...
    def contract_renderer(self): ...
    def public_entrypoints(self): ...
    def environment(self): ...
    def test_command(self): ...
```

Ansible：

```python
class AnsibleProfile(RepoProfile):
    contract_renderer = "ansible_playbook_cli"
```

以后可以扩展：

```text
DjangoProfile
RayProfile
TransformersProfile
...
```

RepoChain 本身保持 repository-agnostic。

---

# 18. 第九项：重新明确 Scoring 的研究定位

当前：

```text
a = λr + (1-λ)p
b = max(0, 1 - 2|p - 0.5|)
score = a × b
```

这里 `p` 同时进入 `a` 和 `b`。

可能出现：

```text
p = 1.0
Solver 很强

但是：
b = 0

最终：
Node Score = 0
```

建议考虑两个实验版本。

## 18.1 Gödel0-HGM

更接近 HGM-style outer loop：

```text
Proposer Quality
=
Eligibility Gate

Solver Utility
=
Parent Selection Utility
```

例如：

```text
Node 进入 Archive 需要：

batch_complete = true
valid_yield >= threshold
causal_ablation_pass >= threshold
difficulty_score >= threshold
```

然后 Parent Selection 使用：

```text
U_i = a_i
```

其中 Proposer score `b` 只作为：

```text
quality gate
或 tie-breaker
```

这种版本最符合：

> HGM，只是 Benchmark Task Provider 被 Evolvable Proposer 替换。

## 18.2 Gödel0-Joint

继续使用：

```text
score = a × b
```

强调：

> Solver 与 Proposer 是对等 Joint Node Components。

建议两个版本都保留用于消融：

```text
Gödel0-HGM
vs
Gödel0-Joint
```

---

# 19. 推荐最终代码架构

```text
Trusted Control Layer
│
├── HGM-style Evolution Loop
│   ├── ParentSelector
│   ├── Archive
│   ├── Budget
│   ├── CycleDiagnosis
│   └── SelfImprovement
│
├── TaskProvider
│   ├── BenchmarkTaskProvider
│   └── ProposerTaskProvider
│
├── TrustedTaskValidator
├── TrustedSolverEvaluator
└── TaskStore


Evolvable Node
│
├── coding_agent.py
├── llm_withtools.py
├── tools/
├── prompts/
│
├── Solver behavior
│
├── Proposer
│   ├── proposer_main.py
│   └── workflows/
│       └── RepoChainWorkflow
│           ├── Weakness Identification
│           ├── Repository Transfer
│           ├── Semantic Chain Discovery
│           ├── Contract Generation
│           ├── Mutation Planning
│           └── Causal Ablation
│
├── Mutation Backends
│   ├── LM Modify
│   ├── LM Rewrite
│   ├── Procedural
│   └── PR Replay
│
├── Repo Profiles
└── Shared Tools
```

核心原则：

```text
RepoChain
=
Workflow

不是 Tool
不是普通 Strategy
```

---

# 20. 推荐修改顺序

## Phase 1：冻结 HGM-compatible Outer Loop

目标：

```text
Task
→ Solver
→ Evaluation
→ Diagnosis
→ Self-Modification
```

接口与 HGM 保持一致。

新增：

```text
TaskProvider
```

将任务来源解耦。

验收：

```text
BenchmarkTaskProvider
和
ProposerTaskProvider

可以无缝接入同一 Evaluator。
```

## Phase 2：RepoChain Workflow Refactor

把：

```text
repo_chain = strategy
```

升级为：

```text
RepoChainWorkflow = Proposer default workflow
```

LM / Procedural / PR Replay 降为内部 Mutation Backend。

## Phase 3：Root Bootstrap

实现：

```text
P_0.bootstrap()
→ T_0
→ S_0(T_0)
```

移除正式实验对：

```text
GODEL0_BOOTSTRAP_SOLVER_TRAJECTORY
```

的依赖。

## Phase 4：HGM-style Special Failure System

真正实现并接通：

```text
Solver special cases
Proposer / RepoChain special cases
Shared-tool / runtime cases
```

## Phase 5：Evidence System

实现：

```text
alert-conditioned evidence selection
success contrast
raw excerpt
bounded context
```

## Phase 6：Task Source Quota

实现：

```text
Parent weakness
+
Current Child weakness
```

的显式来源配额和 metadata。

## Phase 7：RepoProfile

把 Ansible-specific logic 从通用 Planner 搬到：

```text
AnsibleProfile
```

## Phase 8：Scoring Ablation

至少比较：

```text
Gödel0-HGM:
Solver utility + Proposer quality gate

Gödel0-Joint:
a × b
```

## Phase 9：Apptainer / HPC Execution

最后统一接通：

```text
Solver
Proposer
RepoChain
Trusted Validation
Self-Improvement
```

的 Apptainer backend。

在单机流程稳定前，不优先做大规模并发。

---

# 21. P0 修改清单

当前建议最先完成：

```text
P0-1
把 RepoChain 从 SWESmith strategy
提升为 Proposer default workflow。

P0-2
建立 TaskProvider abstraction，
让外层与 HGM 保持一致。

P0-3
实现 Root Proposer bootstrap mode，
去掉外部 trajectory 依赖。

P0-4
真正接通 Solver / Proposer / Tool special detectors。

P0-5
把 Joint Diagnosis Evidence
从 first-N logs 升级为 alert-conditioned evidence。

P0-6
明确 Parent Failure / Current Child Weakness task quota。

P0-7
把 Ansible-specific logic 移入 RepoProfile。
```

完成这七项后，Gödel0 的代码结构与研究故事基本可以重新统一。

---

# 22. 最终不变量

整个系统始终保持：

```text
1. 一个 Node = 一个完整 Agent System。
2. 一棵 Evolution Tree。
3. 不分别选择 Solver Parent / Proposer Parent。
4. 不预先选择 evolve Solver / evolve Proposer。
5. RepoChain 是 Proposer Workflow，不是 Tool。
6. RepoChain 默认存在，但其实现属于 Node，可以 self-evolve。
7. Trusted Controller 不知道 RepoChain 内部实现。
8. Proposer-generated Task 与 Benchmark Task 使用同一 Solver 接口。
9. Agent 可以修改自己如何解题、如何出题、如何使用工具。
10. Agent 不能修改 Trusted Evaluator、Validator 和 TaskStore truth。
11. 每轮 Self-Improvement 只解决一个 primary root cause。
12. HGM-style 系统性失败获得优先诊断机会。
```

---

# 23. 一句话总结

> **不要删掉 RepoChain，而是把它从“一个 SWE-smith strategy”提升为“Gödel0 Proposer 的默认、可进化任务生成 workflow”；同时把 Gödel0 外层重新收束为 HGM-style self-improvement loop，并通过统一 TaskProvider 让唯一核心差异变成：HGM 使用固定 Benchmark Tasks，Gödel0 使用由当前 Node 的 Evolvable RepoChain Proposer 动态生成的 Coding Tasks。**
