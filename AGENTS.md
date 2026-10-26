# Repository instructions for Codex

These instructions apply to the entire repository: application code, experiments, tests, native and upstream sources, adapters, scripts, comments, docstrings, documentation, error messages, identifiers, and filenames.

## Keep history out of the implementation

Source files and tests must describe current behavior, not the conversation or development process that produced them.

Do not introduce or retain references to:

- Private conversations, prompts, reviewer comments, chats, or instructions such as “as discussed” or “the reviewer asked”.
- Task, sprint, review, or finding identifiers such as `Task 5`, `task7`, `FINDINGS #5`, `S0.4`, `S11`, or `R1`.
- Development narratives such as “pre-fix”, “post-fix”, “after this test passes”, “this pass”, “this release”, “landed in commit”, “the workaround is no longer needed”, or “will be added later”.
- Assistant or agent identities in ordinary source, tests, filenames, or comments.
- Change announcements such as “new”, “now”, “currently”, “existing”, “unchanged”, “previously”, or “final” when they merely distinguish a development version from an earlier one. These words are acceptable when they describe actual runtime semantics rather than history.
- A planned or hypothetical feature described as though it already exists. Describe implemented behavior and actual limitations only.

Do not turn comments into a changelog. Version control and explicitly requested change logs are the places for development history.

When cleaning up, inspect existing text rather than only checking new edits. Remove or rewrite every violation in scope. Deleting a stale comment is usually better than replacing it with another narrative.

## Use durable names

Filenames, directories, functions, classes, variables, tests, and error messages must have durable domain meanings.

- Do not encode task numbers, review rounds, fix attempts, release stages, assistant names, or private plan names in names.
- Prefer names such as `governed_flow.py`, `verify_publication`, or `test_rejects_duplicate_evidence` over names such as `task5_flow.py`, `new_verify`, `test_post_fix`, or `final_cleanup.py`.
- Do not use `final`, `new`, `updated`, `v2`, `fix`, `cleanup`, `patched`, or `copy` as historical qualifiers in filenames. A domain term or an actual artifact version is acceptable when it has a defined technical meaning.
- Rename historical identifiers encountered during cleanup, including test names and internal helper names. Check references after renaming.
- Preserve public APIs, wire formats, protocol constants, and external compatibility. If a public historical name must be retired, use a documented compatibility path rather than breaking callers merely for cosmetic reasons.
- Do not rename preserved measurement runs or alter measurement data to satisfy these rules.

## Write comments that earn their place

Comments and docstrings should explain current contracts, invariants, security boundaries, units, ownership, lifecycle, protocol requirements, or non-obvious reasons for an implementation choice.

Do not write comments that:

- Restate the next line of code or repeat the function name.
- Narrate what was changed, fixed, added, removed, refactored, or verified during development.
- Refer to what another agent, reviewer, test, or previous implementation “caught” or “missed”.
- Claim that code is obviously correct, fully safe, comprehensive, production-ready, or guaranteed without precise supporting conditions.
- Describe a test as passing or a bug as fixed instead of stating the behavior the test checks.
- Use throat-clearing or filler such as “Note that”, “It is important to remember”, “This is the point of”, or “In order to”.

Prefer direct technical statements. For example, write “Child frames receive independent native copies before mutation” rather than “The new copy-on-write fix prevents the sibling mutation bug found in review”.

## Avoid AI-style prose and unsupported claims

Use plain, specific language. Remove filler, hype, false symmetry, generic transitions, and performative certainty.

- Avoid “seamless”, “robust”, “powerful”, “cutting-edge”, “leverages”, “delve”, “elevate”, “unlock”, “game-changing”, and similar promotional wording unless quoting a defined external term.
- Do not use “simply”, “just”, “obviously”, “clearly”, “of course”, or “trivially” to dismiss complexity.
- Do not pad text with “In today’s world”, “It is worth noting”, “Furthermore”, “Moreover”, or repetitive summary sentences that add no information.
- Do not claim “unchanged”, “identical”, “independent”, “complete”, “all”, “never”, or “always” without checking the relevant evidence and scope.
- Do not claim a performance improvement without measurements. Complexity analysis and benchmark results must be presented as different kinds of evidence.
- For measurements, identify the exact timed operation, inclusions and exclusions, repetitions, aggregation, baseline, and whether model startup or inference is included.
- Distinguish ordinary ungoverned processing from comparisons between two governed implementations.
- Use “offline” for a verifier that runs without publisher or network state; use “independent” only when the relevant implementation and trust independence are actually established.

## Required cleanup audit

For a comment, documentation, naming, or AI-slop cleanup task, scan the whole repository, including native and test code. Do not restrict the audit to files already edited. Exclude only VCS metadata, dependency environments, caches, and build output.

Use this command as an audit lead on systems with `rg`:

```text
rg -n --hidden -g '!.git/**' -g '!.venv/**' -g '!**/target/**' -g '!**/__pycache__/**' -i 'break caught|findings|pre[-_ ]?fix|post[-_ ]?fix|task[-_ ]?[0-9]+|as discussed|previous conversation|this (pass|release|commit|submission)|after this (change|fix|test)|landed|seamless|robust|leverages?|delve|obviously|simply|unchanged|final|new|updated|cleanup' .
```

Inspect every hit in context. A search hit is not permission for blind replacement: protocol versions, runtime state, legitimate domain terminology, preserved results, and explicitly requested change logs may be valid. A clean search result alone is not sufficient; also read the files being changed and look for commentary whose meaning depends on unavailable context.

## Completion checklist

Before finishing a cleanup or review:

1. Search the whole requested scope, not only the diff.
2. Remove historical and conversational commentary from code, tests, errors, names, and documentation.
3. Verify that every remaining comment describes current behavior and adds technical information.
4. Verify that names no longer encode development history and that renamed references are updated.
5. Verify factual claims against implementation and retained evidence; qualify anything not established.
6. Preserve behavior, public contracts, and immutable measurement data unless a change is explicitly requested.
7. Avoid unrelated mass reformatting or line-ending churn.
8. Run focused tests and substantive lint checks after behavioral changes. Report unrun tests and environment limitations precisely.
