# LM Modify Bug Introduction

You are a software developer doing chaos monkey testing.
Your job is to rewrite a function such that it introduces a logical bug that will break existing unit test(s) in a codebase.

To this end, some kinds of bugs you might introduce include:
- Alter calculation order for incorrect results: Rearrange the sequence of operations in a calculation to subtly change the output (e.g., change (a + b) * c to a + (b * c)).
- Introduce subtle data transformation errors: Modify data processing logic, such as flipping a sign, truncating a value, or applying the wrong transformation function.
- Change variable assignments to alter computation state: Assign a wrong or outdated value to a variable that affects subsequent logic.
- Mishandle edge cases for specific inputs: Change handling logic to ignore or improperly handle boundary cases, like an empty array or a null input.
- Modify logic in conditionals or loops: Adjust conditions or loop boundaries (e.g., replace <= with <) to change the control flow.
- Introduce off-by-one errors in indices or loop boundaries: Shift an index or iteration boundary by one, such as starting a loop at 1 instead of 0.
- Adjust default values or constants to affect behavior: Change a hardcoded value or default parameter that alters how the function behaves under normal use.
- Reorder operations while maintaining syntax: Rearrange steps in a process so the function produces incorrect intermediate results without breaking the code.
- Swallow exceptions or return defaults silently: Introduce logic that catches an error but doesn't log or handle it properly, leading to silent failures.

Tips about the bug-introducing task:
- It should not cause compilation errors.
- It should not be a syntax error.
- It should be subtle and challenging to detect.
- It should not modify the function signature.
- It should not modify the documentation significantly.
- For longer functions, if there is an opportunity to introduce multiple bugs, please do!
- Please DO NOT INCLUDE COMMENTS IN THE CODE indicating the bug location or the bug itself.
- Return exactly one Python code block under `Bugged Code`.
- If a target symbol is provided, return the complete modified definition for that symbol, including decorators, signature, and body.
- Do not return unrelated functions, partial snippets, or surrounding file content unless the target is the entire file.
- The returned code must make at least one behavioral change compared with the input.

Your answer should be formatted as follows:

Explanation:
<explanation>

Bugged Code:
```
<bugged_code>
```

## Target

- File: {target_file}
- Symbol: {target_symbol}

## Current Source

```python
{source}
```

## Desired Behavior

{desired_behavior}

Apply the edit directly to the file in the workspace.
