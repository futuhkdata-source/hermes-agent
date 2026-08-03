# P1-B Execution Advisory Controller

**Status:** candidate design; offline only; production activation is out of scope.

## Goal

Turn the already-approved `hermes.execution-shadow.v1` metadata into deterministic, privacy-minimal closure advice without changing agent behaviour.

## Non-negotiable boundary

P1-B does not add a hook, model tool, prompt block, route, scheduler, worker, or control-flow return value. It does not change the P1-A snapshot/event schema or current runtime configuration. The first deliverable is a pure evaluator plus an offline replay command.

Every advisory result declares these effects as `false`: `block`, `stop`, `rewrite`, `route`, `schedule`, and `spawn`.

## Input and output

- Input: one parsed `hermes.execution-shadow.v1` snapshot.
- Output: `hermes.execution-advisory.v1` metadata only.
- Correlation: a bounded SHA-256 digest; raw session/task/turn identifiers are not copied.
- No user/assistant text, child goals, tool arguments/results, or exception messages are accepted into output fields.
- Offline aggregate: `hermes.execution-advisory-replay.v1`; it reports only counts by finite action, urgency, and reason code.

## Finite advisory actions

Ordered by precedence:

1. `REVIEW_TERMINAL_OUTCOME`
2. `REVIEW_STALE_TURN`
3. `VERIFY_EXECUTION_PROOF`
4. `FINALIZE_ONLY`
5. `MOVE_TO_CLOSURE`
6. `FREEZE_SCOPE`
7. `NO_ACTION`

Multiple actions may be present; `primary_action` is the highest-precedence action. Reason codes are finite and deterministic.

## Decision contract

- Failed, interrupted, incomplete, exceptional, or iteration-exhausted terminal turns require terminal review.
- A non-terminal snapshot older than the explicit replay cutoff requires stale-turn review.
- A passive durable-running claim without execution proof requires proof verification, including after terminal completion.
- While a turn is non-terminal, finalization budget or reviewer/remediation/finalization-work limit warnings produce `FINALIZE_ONLY`.
- While a turn is non-terminal, closure budget produces `MOVE_TO_CLOSURE`.
- While a turn is non-terminal, scope-freeze budget or post-freeze discovery/helper/rework warnings produce `FREEZE_SCOPE`.
- Once a turn has completed successfully, prior phase-budget and rework signals are retrospective and do not create stale `FREEZE_SCOPE`/`MOVE_TO_CLOSURE`/`FINALIZE_ONLY` actions.
- A successful terminal turn with no unproven durable claim produces `NO_ACTION`.

## Acceptance gates for this candidate

- RED/GREEN tests for every boundary and precedence rule.
- Existing P1-A 22-test contract remains byte-behaviour compatible.
- Replay all live snapshots with zero invalid records and no identifier leakage.
- Determinism: identical snapshot + `as_of` yields identical output bytes.
- Fail-open integration boundary: this candidate is not imported by the live plugin.
- One immutable staged diff, one independent read-only GO/NO-GO review, at most one remediation cycle.

## Offline replay command

```bash
python plugins/observability/execution-shadow/replay.py \
  --telemetry-root ~/.hermes/telemetry/execution-shadow \
  --as-of 2026-08-03T17:33:57+00:00
```

The replay clock is mandatory so the result is reproducible. Exit `0` means every snapshot was valid; exit `2` means the input directory is missing or at least one record is invalid. Standard output contains only aggregate JSON and never paths, identifiers, or row contents.

## Explicitly deferred

- Importing the evaluator from live hooks.
- Writing advisory sidecars during live turns.
- Displaying advice to the agent or user.
- Any enforcement or automated action.
- Changes to prompt, routing, durable execution, orchestration, Kanban, cron, or Gateway lifecycle.
