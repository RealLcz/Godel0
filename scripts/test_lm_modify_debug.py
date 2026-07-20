#!/usr/bin/env python3
"""Quick test of LM Modify generate() with vLLM."""
import sys, os
sys.path.insert(0, "src")
sys.path.insert(0, "initial_agent/src")

from swesmith.lm_modify import LMModify
from swesmith.engine import BugGenerationPlan, RepoSpec, BugConstraints

class VLLMAgentAdapter:
    def __init__(self, host, port, model):
        import openai
        self.client = openai.OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")
        self.model = model
    def chat(self, system_prompt, user_prompt, temperature=1, max_tokens=16384):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

host = os.environ.get("VLLM_HOST", "127.0.0.1")
port = os.environ.get("VLLM_PORT", "8000")
model = "Qwen/Qwen3.6-35B-A3B"

adapter = VLLMAgentAdapter(host, port, model)
lm = LMModify(agent_adapter=adapter)

repo_spec = RepoSpec(
    repo_id="ansible",
    repo_path="repo_pool/ansible",
    base_commit="abc",
    test_command="pytest",
)

# Test on helpers.py (small file)
plan = BugGenerationPlan(
    plan_id="test_lm_001",
    target_repo_id="ansible",
    target_base_commit="abc",
    target_file="lib/ansible/utils/helpers.py",
    target_symbol="pct_to_int",
    strategy="lm_modify",
    operator="lm_modify",
    constraints=BugConstraints(max_modified_lines=15, desired_behavior="Introduce a subtle bug"),
    seed=42,
)

print("=== Calling LMModify.generate() ===")
candidates = lm.generate(plan=plan, node_code_dir="/tmp", repo_spec=repo_spec, output_dir="/tmp/lm_test")
print(f"\nCandidates: {len(candidates)}")
for c in candidates:
    print(f"  ID: {c.candidate_id}")
    print(f"  Patch length: {len(c.bug_patch)} chars")
    print(f"  Patch:")
    print(c.bug_patch[:500])
    print(f"  Explanation: {c.generation_metadata.get('explanation', '')[:200]}")
