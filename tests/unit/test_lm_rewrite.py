from initial_agent.src.swesmith.entity_index import EntityIndex
from initial_agent.src.swesmith.lm_rewrite import LMRewrite


def _entity_for(source: str, name: str):
    index = EntityIndex()
    index.index_file("sample.py", source=source)
    return index.find(name)[0]


def test_lm_rewrite_preserves_surrounding_file_text_for_body_only_output():
    source = '''# keep this comment

def helper():
    return "helper"


def target(x=1):
    """doc"""
    y = x + 1
    return y


def after():
    return "after"
'''
    entity = _entity_for(source, "target")
    rewritten = LMRewrite()._replace_entity_body(source, entity, "return x - 1")

    assert "# keep this comment" in rewritten
    assert 'return "helper"' in rewritten
    assert 'return "after"' in rewritten
    assert '"""doc"""' in rewritten
    assert "return x - 1" in rewritten
    assert "y = x + 1" not in rewritten


def test_lm_rewrite_accepts_complete_function_output_without_nesting_it():
    source = """def target(x, y=2):
    return x + y
"""
    entity = _entity_for(source, "target")
    generated = """def target(x, y=2):
    total = x - y
    return total
"""
    rewritten = LMRewrite()._replace_entity_body(source, entity, generated)

    assert rewritten.count("def target") == 1
    assert "total = x - y" in rewritten
    assert "return x + y" not in rewritten


def test_lm_rewrite_blanks_only_target_function_body():
    source = """def first():
    return 1


def target(value):
    if value:
        return value
    return 0
"""
    entity = _entity_for(source, "target")
    blanked = LMRewrite()._blank_out_function(source, entity)

    assert "def first" in blanked
    assert "return 1" in blanked
    assert "def target(value):" in blanked
    assert "pass" in blanked
    assert "if value" not in blanked
