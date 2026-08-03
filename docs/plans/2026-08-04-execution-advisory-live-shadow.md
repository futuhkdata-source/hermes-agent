# P1-C Live Shadow Advisory Activation

**Status:** default-off candidate implemented; independent review and production cutover pending.

## Purpose

Import the already-verified pure `hermes.execution-advisory.v1` evaluator into the live execution-shadow plugin and persist its latest recommendation as local profile-scoped telemetry. Advice remains invisible to the model and user and cannot influence control flow.

## Activation gate

Add two keys under `execution_shadow`:

```yaml
advisory_enabled: false
max_advisory_file_bytes: 65536
```

- Both source and `DEFAULT_CONFIG` remain default-off.
- Missing, malformed, or non-boolean `advisory_enabled` resolves to `false`.
- The byte cap is clamped to `[4096, 1000000]`.
- The existing `enabled: true` and `mode: shadow` gates remain mandatory.
- The config is profile-local and is re-read through the existing mtime/size cache, so the advisory gate can be changed after a dark-load restart without another restart.

## Live write contract

After the P1-A snapshot and event are persisted, and only when the advisory gate is enabled:

1. Evaluate the exact in-memory P1-A snapshot with no implicit replay clock.
2. Serialize only the existing `hermes.execution-advisory.v1` result.
3. Atomically replace `telemetry/execution-shadow/advisories/<turn_digest>.json`.
4. Require the serialized result to fit the configured byte cap.
5. Keep the advisory directory `0700` and files `0600` on POSIX.
6. Clean temporary files after both success and failure.

The file is the latest live advisory for one turn, not an append-only history. Replacement is its bound; rotation does not apply to an individual sidecar. Dataset retention remains aligned with the existing P1-A per-turn telemetry lifecycle.

## Fail-open and control boundary

- Evaluator import failure disables the plugin at load and is caught by normal PluginManager isolation during dark-load verification.
- Evaluation, size validation, and advisory write failures are caught inside the advisory branch; the P1-A snapshot/event already persisted and future hooks continue.
- No hook is added or removed.
- Every hook continues to return `None`.
- No message, prompt, tool, route, scheduler, worker, durable task, Kanban state, agent state, or control-flow result is created or modified.
- Sidecars are never injected into conversation context and are not delivered externally.

## Privacy contract

Sidecars may contain only finite action/reason codes, urgency, bounded numeric/boolean observations, a 24-character digest, schema names, and six explicit `false` control-effect flags. They must not contain session/task/turn IDs, platform IDs, paths, prompts, responses, child goals, tool arguments/results, or dynamic exception text.

## Cutover sequence

1. RED/GREEN tests with advisory default-off.
2. Sandbox/profile-isolation and adversarial write tests.
3. Focused and broad regressions, lint, compile, privacy, permissions, and performance.
4. Freeze exact staged bytes and run one independent read-only review.
5. Commit exact GO bytes while live config remains advisory-off.
6. Create restrictive config backup and rollback manifest.
7. Use a restart-safe external owner to restart Gateway with advisory still off (dark load).
8. Verify Gateway/API Server/Feishu health, plugin hooks, zero new sidecars, and config provenance.
9. Enable only `execution_shadow.advisory_enabled` and run one local API Server canary through the live Gateway process, using Bearer auth when the active server is configured with a key and the existing localhost-only contract otherwise.
10. Verify a valid sidecar, unchanged response path, permissions, no leakage, and stable Gateway health.
11. On any failure, restore the config backup (or force advisory false), verify health, and issue `NO-GO`.

## Explicitly out of scope

- Advice display or prompt/context injection.
- Automated action or enforcement.
- New hooks, tools, routes, workers, schedulers, cron business logic, orchestration, Kanban, or durable claims.
- Advisory transition history or external delivery.
- Remote push/PR unless separately approved.
