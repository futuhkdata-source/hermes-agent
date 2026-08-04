# P1-C.1 Bounded Advisory Decision-Transition Audit

**Status:** live with `GO_P1C1_TRANSITION_AUDIT` on 2026-08-04 HKT; source commit `ab8df39efa0b669e9c9c92cf3a07d509966788e1`.

## Purpose

Preserve when a live shadow advisory decision changes without exposing advice to the model/user or granting it any control authority. P1-C latest-per-turn sidecars remain byte-compatible and continue to be the current-state view; P1-C.1 adds a separate, bounded calibration history.

## Config contract

Add four profile-scoped keys under `execution_shadow`:

```yaml
advisory_transition_log_enabled: false
max_advisory_transition_record_bytes: 65536
max_advisory_transition_total_bytes: 8000000
max_advisory_transition_records: 1024
```

- The transition gate is independently default-off and is effective only when `enabled`, `mode: shadow`, and `advisory_enabled` are already valid.
- Missing, malformed, or non-boolean transition enablement resolves to `false`.
- Per-record bytes are clamped to `[4096, 1000000]`.
- Total retained bytes are clamped to `[64000, 100000000]`.
- Retained records are clamped to `[1, 4096]`.
- Existing config mtime/size caching must observe a gate change without a second module reload.

## Decision-transition semantics

The canonical decision vector is:

1. `primary_action`;
2. ordered `actions`;
3. ordered `reason_codes`;
4. `urgency`.

A transition is stored only when this vector differs from the last successfully stored vector for that in-memory turn. Numeric/boolean changes in `observed` alone do not create records. The first stored decision for a turn has `previous_primary_action: null`. After a Gateway restart, the first observed decision may be stored again; `decision_digest` allows bounded offline deduplication.

The previous signature advances only after the immutable record has been file-synced, published without overwrite, and the containing directory has been synced. Oversize, unsafe-path, serialization, write, file-sync, close, publish, or directory-sync failure leaves it unchanged so a later hook can retry. Retention pruning happens only after commit; pruning failure is logged fail-open and the next real transition retries it without duplicating the committed decision.

## Record schema

Each immutable JSON record has exactly:

```json
{
  "schema_version": "hermes.execution-advisory-transition.v1",
  "mode": "advisory-transition",
  "source_schema_version": "hermes.execution-shadow.v1",
  "timestamp": "timezone-aware source updated_at",
  "turn_digest": "24 lowercase hex",
  "decision_digest": "24 lowercase hex",
  "previous_primary_action": null,
  "primary_action": "finite action code",
  "actions": ["finite action codes"],
  "reason_codes": ["finite reason codes"],
  "urgency": "none|low|medium|high",
  "observed": "the bounded metadata-only advisory observation object",
  "control_effects": {
    "block": false,
    "rewrite": false,
    "route": false,
    "schedule": false,
    "spawn": false,
    "stop": false
  }
}
```

No session/task/turn/platform IDs, paths, prompt/response text, child goals, tool arguments/results, credentials, or dynamic exception strings are permitted.

## Secure immutable storage

- Path: `telemetry/execution-shadow/advisory-transitions/transition-YYYYMMDD-<time_ns>-<decision_digest>-<random>.json`.
- Directory mode `0700`; lock and record files `0600` on POSIX.
- Open the profile telemetry root with `O_DIRECTORY|O_NOFOLLOW`, create/open the transition directory relative to that root descriptor, and perform all subsequent file operations relative to the opened directory descriptor.
- Reject a symlinked transition directory. A later path swap cannot redirect writes because the open descriptor remains anchored to the original directory inode.
- Serialize one exact bounded record, write it to an `O_CREAT|O_EXCL|O_NOFOLLOW` private temporary file, `fsync` it, and close it before publication.
- Publish with a hard-link operation that fails on destination collision; retry a randomized name rather than overwriting any existing record. Remove the temporary name so the final record has link count one.
- `fsync` the directory before reporting success. If directory sync fails, remove the just-published name and fail without advancing the in-memory signature.
- Use a private `.transition.lock`, validate `(st_dev, st_ino)`, regular-file type, and link count, and take `flock(LOCK_EX)` on POSIX. This serializes publish and retention across Gateway/process overlap.
- Scan only matching immutable record names, no-follow stat them, require regular files with link count one, and cap directory scanning at 16,384 entries.
- After a committed publish, prune oldest eligible records until both configured count and total-byte bounds are met; never follow or modify nonmatching/symlink/device entries.

Immutable one-record segments deliberately replace the original JSONL active-file proposal. This removes partial-append rollback, active-file rotation, rotated-name overwrite, and cross-PID active-file pruning classes while retaining a bounded chronological audit history.

## Ordering and fail-open boundary

For each hook:

1. persist the P1-A snapshot;
2. append the P1-A event;
3. evaluate and atomically persist the P1-C latest sidecar;
4. if enabled and the decision changed, commit one immutable P1-C.1 transition record.

Any P1-C.1 failure occurs after P1-A and P1-C persistence and is caught inside the advisory branch. Hooks still return `None`; no message, prompt, tool, route, scheduler, worker, agent/task state, orchestration, Kanban, or control-flow behavior changes.

## Review history and bounded remediation

The first immutable review bundle's predecessor used active JSONL files. The independent reviewer returned NO-GO because of a symlinkable transition directory, path-based TOCTOU, missing fsync durability, incomplete append/rotation rollback, collision-overwrite risk, and process-local retention coordination. The single bounded remediation replaced that substrate with the secure immutable design above and added one-to-one RED tests for:

- symlinked directories, hardlinked lock files, and directory path swaps;
- partial write, file-sync, directory-sync, and close failure cleanup;
- collision preservation with no-overwrite retry;
- bounded count/total-byte retention;
- cross-process flock and concurrent retention;
- day rollover, privacy, permissions, hot config, profile isolation, and unchanged P1-C sidecars.

No second reviewer/remediation loop is permitted; the owner performs one focused deterministic re-check and issues final GO/NO-GO.

## Verification and cutover

1. Focused P1-A/B/C/C.1 contracts, full plugin suite, PluginManager tests, replay, lint, compile, privacy, permissions, multiprocessing, and performance.
2. Freeze remediated exact bytes and verify the staged patch against its manifest.
3. Commit while the transition gate remains off.
4. Create a config rollback that preserves the already-live P1-C advisory gate.
5. Restart-safe dark load; prove existing sidecars continue while no transition record is created.
6. Enable only the transition gate and run one Bearer-authenticated API Server canary using the owner-only profile credential.
7. Verify one valid immutable transition, stable Gateway/Feishu/API Server health, and unchanged control boundaries.
8. On failure, restore the config backup, leaving P1-C live and P1-C.1 off.

## Live activation evidence

- Final verdict: `GO_P1C1_TRANSITION_AUDIT` at 2026-08-04 05:18 HKT.
- Source commit: `ab8df39efa0b669e9c9c92cf3a07d509966788e1`.
- Restart proof: Gateway PID `186483` → `191606`.
- Dark canary: P1-C sidecar created while P1-C.1 created zero transition records.
- Active canary: one 674-byte `hermes.execution-advisory-transition.v1` immutable record with `NO_ACTION`; marker observed in the API response but absent from stored metadata.
- Live boundaries: P1-C advisory and P1-C.1 transition gates are true; all six control effects are false; Gateway, Feishu, and API Server are connected.
- Rollback config: `/home/ubuntu/.hermes/backups/p1c1-transition-cutover-20260804-051033-HKT/config.before.yaml`.
- Cutover evidence: `/home/ubuntu/.hermes/backups/p1c1-transition-cutover-20260804-051033-HKT/cutover-result.json`.

## Explicitly out of scope

- Advice display or delivery.
- Prompt/context injection.
- Automated action or enforcement.
- New hooks, tools, routes, workers, schedulers, orchestration, Kanban, or durable claims.
- Remote push/PR unless separately approved.
