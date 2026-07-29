# Shadow Execution Controller Implementation Plan

> **For Hermes:** Execute this plan directly with strict TDD; reserve one independent reviewer for the frozen pre-commit snapshot.

**Goal:** Add an opt-in, default-profile-only shadow controller that records execution phase, iteration-budget gates, helper/review/remediation counts, completion evidence, and contract warnings without changing agent behavior.

**Architecture:** Ship a standalone bundled plugin at `plugins/observability/execution-shadow/`. It consumes existing lifecycle hooks and writes privacy-minimal, profile-scoped local telemetry. Add only the missing observational metadata to existing hook payloads; callbacks are fail-open and never return control-flow values.

**Tech Stack:** Python stdlib, Hermes plugin hooks, YAML config, pytest.

---

### Task 1: Lock the plugin and telemetry contract

**Files:**
- Create: `tests/plugins/test_execution_shadow_plugin.py`
- Create: `plugins/observability/execution-shadow/plugin.yaml`

**Steps:**
1. Write failing tests for manifest/discovery and explicit opt-in.
2. Write failing tests for profile-scoped `execution_shadow.enabled` gating.
3. Require telemetry to contain counters/booleans only—never user text, assistant text, tool arguments, tool results, or child goals.
4. Run the focused test and confirm RED because the plugin does not exist.

### Task 2: Lock phase and budget observations

**Files:**
- Modify: `tests/plugins/test_execution_shadow_plugin.py`
- Create: `plugins/observability/execution-shadow/__init__.py`

**Steps:**
1. Add failing tests for discovery → implementation → verification → closure → finalization → terminal monotonic phases.
2. Add failing boundary tests at 60%, 75%, and 85% using runtime-supplied `budget_used` / `budget_max`.
3. Implement the minimum in-memory state and profile-scoped snapshot/event writer.
4. Verify GREEN.

### Task 3: Observe helpers, reviews, remediation, and completion evidence

**Files:**
- Modify: `tests/plugins/test_execution_shadow_plugin.py`
- Modify: `plugins/observability/execution-shadow/__init__.py`

**Steps:**
1. Add failing tests for helper, reviewer, and remediation counters from `subagent_start` child goals.
2. Add failing tests for warnings after scope freeze/finalization and for limits above one reviewer/remediation.
3. Add failing tests for successful test, commit, delivery, rollback, final response, and GO/NO-GO evidence.
4. Implement privacy-minimal classifiers; discard raw content immediately after classification.
5. Verify GREEN, including concurrent-turn isolation and fail-open I/O failure.

### Task 4: Supply accurate runtime metadata through existing hooks

**Files:**
- Modify: `agent/conversation_loop.py`
- Modify: `agent/turn_finalizer.py`
- Test: focused run-agent / turn-finalizer hook tests

**Steps:**
1. Add failing tests that `pre_api_request` / `post_api_request` expose `budget_used` and `budget_max`.
2. Add failing tests that `post_llm_call` exposes `completed`, `failed`, `interrupted`, `turn_exit_reason`, `api_call_count`, and budget values.
3. Add the metadata only; do not alter loop conditions, messages, tools, output, routing, or prompt construction.
4. Run focused tests and prove behavior parity.

### Task 5: Default config surface and controlled activation

**Files:**
- Modify: `hermes_cli/config.py`
- Modify: local `/home/ubuntu/.hermes/config.yaml` only after review GO

**Steps:**
1. Add failing config-default tests for `mode: shadow`, disabled-by-default, and 0.60/0.75/0.85 thresholds.
2. Add defaults without changing existing user config behavior.
3. Before production activation, back up the live config and record hashes.
4. Enable `observability/execution-shadow` plus `execution_shadow.enabled=true` only in the default profile.

### Task 6: Verification, review, and cutover

**Steps:**
1. Run the plugin tests alone, relevant hook/config suites, then the bounded full suite.
2. Run offline fresh-agent probes and verify output equality with the feature disabled vs enabled.
3. Freeze/stage exact bytes and obtain one independent read-only GO/NO-GO review.
4. If GO, commit exact bytes, restart Gateway through the external one-shot pattern, and run one real default turn.
5. Verify telemetry under `~/.hermes/telemetry/execution-shadow/`, no department-profile output, no active durable/Kanban work, and cleanup temporary cutover assets.
6. Report P1-A GO/NO-GO. Do not enable P1-B enforcement in this scope.
