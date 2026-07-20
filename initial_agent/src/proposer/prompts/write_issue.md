# Generate GitHub-style Issue Draft

You are a software engineer helping to create a realistic dataset of synthetic GitHub issues.

You will be given the following input:

1. Patch: A git diff output/pull request changes that introduces a bug (included in the <patch> tag).
2. Test output: The output of running the tests after the patch is applied (included in the <test_output> tag).
3. Test source code: Source code for one or more tests that failed (included in the <test_source_code> tag).

Output: A realistic GitHub issue for the patch.

Guidelines:

- DO NOT explain the fix/what caused the bug itself, focus on how to reproduce the issue it introduces.
- Do not mention pytest or what exact test failed. Instead, generate a realistic issue.
- If possible, include information about how to reproduce the issue. An ideal reproduction script should raise an error
  or print an unexpected output together with the expected output.
- DO NOT GIVE AWAY THE FIX! THE SOLUTION CODE SHOULD NEVER APPEAR IN YOUR RESPONSE.
- DO NOT SAY THAT EXISTING TEST(s) FAILED.
- DO NOT SUGGEST RUNNING ANY TESTING COMMANDS (e.g., pytest).

<patch>
{patch}
</patch>

<test_output>
{test_output}
</test_output>

<test_source_code>
{test_source_code}
</test_source_code>

**Issue Text**

<START WRITING>
