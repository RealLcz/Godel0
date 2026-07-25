"""Split leading ``KEY=VALUE`` env assignments from a shell-style command.

Repo profiles express test commands in shell syntax, e.g.::

    PYTHONPATH=lib:test/lib python3.11 -m pytest -p no:cacheprovider ...

Trusted runners (CandidateValidator, BackendAwareTestRunner) execute
commands as argv through an ExecutionBackend — without a shell — so the
env prefix must be extracted and passed via the backend ``env`` parameter.
Passing the raw string to ``shlex.split`` made ``PYTHONPATH=lib:test/lib``
the executable name, every trusted run failed with returncode -1, and all
candidates were rejected as ``clean_tests_unusable``.
"""

from __future__ import annotations

import re
import shlex
from typing import Dict, List, Sequence, Tuple, Union

_ENV_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def split_env_assignments(
    command: Union[str, Sequence[str]],
) -> Tuple[Dict[str, str], List[str]]:
    """Return ``(env, argv)`` with leading env assignments removed.

    ``command`` may be a shell-style string or an argv list. Only leading
    tokens of the form ``NAME=value`` are treated as env assignments,
    mirroring POSIX shell simple-command semantics.
    """
    if isinstance(command, str):
        parts = shlex.split(command)
    else:
        parts = [str(part) for part in command]
    env: Dict[str, str] = {}
    index = 0
    while index < len(parts) and _ENV_TOKEN.match(parts[index]):
        key, _, value = parts[index].partition("=")
        env[key] = value
        index += 1
    return env, parts[index:]


__all__ = ["split_env_assignments"]
