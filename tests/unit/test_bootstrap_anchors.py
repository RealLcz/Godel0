"""P0-1: Root bootstrap must resolve real repository anchors."""
from __future__ import annotations

from pathlib import Path

from swesmith.repo_level import declared_target_files
from proposer.code_locator import CodeLocator, RepoIndex
from proposer.workflows.repo_chain.bootstrap import (
    BOOTSTRAP_CAPABILITY_PRIOR,
    build_bootstrap_plans,
)


def _write_mini_repo(root: Path) -> Path:
    """Create a tiny repo with enough symbols for CodeLocator."""
    files = {
        "pkg/parser/tokenize.py": (
            "def tokenize(text):\n"
            "    return text.split()\n"
            "\n"
            "def parse(tokens):\n"
            "    return {'tokens': tokens}\n"
        ),
        "pkg/parser/executor.py": (
            "from pkg.parser.tokenize import tokenize\n"
            "\n"
            "def execute(ast):\n"
            "    return ast\n"
        ),
        "pkg/inventory/manager.py": (
            "class InventoryManager:\n"
            "    def load(self, path):\n"
            "        return {'hosts': []}\n"
            "\n"
            "    def get_host(self, name):\n"
            "        return name\n"
        ),
        "pkg/inventory/host.py": (
            "class Host:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n"
        ),
        "pkg/loader/state.py": (
            "def load_state(path):\n"
            "    return {'path': path}\n"
            "\n"
            "def save_state(data):\n"
            "    return True\n"
        ),
        "pkg/config/options.py": (
            "def load_config(path):\n"
            "    return {}\n"
            "\n"
            "def merge_defaults(cfg):\n"
            "    return cfg\n"
        ),
        "pkg/errors/handler.py": (
            "class ErrorHandler:\n"
            "    def report(self, err):\n"
            "        return str(err)\n"
        ),
        "pkg/plugin/loader.py": (
            "def load_plugin(name):\n"
            "    return name\n"
        ),
        "pkg/pipeline/executor.py": (
            "def run_pipeline(stages):\n"
            "    return list(stages)\n"
        ),
        "pkg/compat/legacy.py": (
            "def legacy_call(x):\n"
            "    return x\n"
        ),
    }
    for rel, src in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(src, encoding="utf-8")
    return root


class _Spec:
    def __init__(self, repo_dir: str, repo_id: str = "toy"):
        self.repo_id = repo_id
        self.repo_dir = repo_dir
        self.base_commit = "deadbeef"


class TestBootstrapRealAnchors:
    def test_plans_set_real_target_file_and_symbol(self, tmp_path: Path):
        repo = _write_mini_repo(tmp_path / "repo")
        index = RepoIndex.build("toy", str(repo), source_dirs=["."])
        plans = build_bootstrap_plans(
            BOOTSTRAP_CAPABILITY_PRIOR,
            _Spec(str(repo)),
            target_count=10,
            max_plans=15,
            repo_index=index,
            code_locator=CodeLocator(),
        )
        assert len(plans) >= 5
        for plan in plans:
            assert plan.target_file, "bootstrap plan missing target_file"
            assert plan.target_symbol, "bootstrap plan missing target_symbol"
            assert plan.target_files, "bootstrap plan missing target_files"
            assert plan.target_file in plan.target_files
            # Files must actually exist in the indexed repo.
            assert (repo / plan.target_file).is_file()

    def test_declared_target_files_nonempty_for_related_files(self, tmp_path: Path):
        repo = _write_mini_repo(tmp_path / "repo")
        index = RepoIndex.build("toy", str(repo), source_dirs=["."])
        plans = build_bootstrap_plans(
            ["cross_file_localization"],
            _Spec(str(repo)),
            target_count=3,
            max_plans=6,
            repo_index=index,
            code_locator=CodeLocator(),
        )
        assert plans
        declared = declared_target_files(plans[0])
        assert declared
        assert plans[0].target_file in declared

    def test_diversity_avoids_duplicate_real_files(self, tmp_path: Path):
        repo = _write_mini_repo(tmp_path / "repo")
        index = RepoIndex.build("toy", str(repo), source_dirs=["."])
        plans = build_bootstrap_plans(
            BOOTSTRAP_CAPABILITY_PRIOR,
            _Spec(str(repo)),
            target_count=8,
            max_plans=12,
            repo_index=index,
            code_locator=CodeLocator(),
        )
        files = [p.target_file for p in plans]
        assert len(files) == len(set(files)), files

    def test_without_repo_returns_empty(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist"
        plans = build_bootstrap_plans(
            BOOTSTRAP_CAPABILITY_PRIOR,
            _Spec(str(missing)),
            target_count=10,
        )
        assert plans == []
