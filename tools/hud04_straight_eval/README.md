# Bounded straight-line IsaacLab evaluation

This directory is an **inference-only diagnostic**, based on the repository's
training revision `3e444244d41febad2e2827168c178f6f68d04ea4`.
It contains no checkpoints, exported policies, robot assets, account credentials,
private artifact URLs, or provisioning commands. Do not add those to Git.

## Validation status

- Pure-Python identity checks, drive comparisons, signed-motion metrics, canonical
  report evidence, and supervisor success/failure/watchdog paths have CPU tests.
- These tests **do not prove simulator compatibility**. No IsaacLab/PhysX rollout
  has been run by this code yet. The first remote execution is a bounded smoke
  test plus a maximum 30-second simulated rollout, not a validated training job.
- The initial candidate environment is IsaacSim 4.5 / IsaacLab 2.1.0 / Python 3.10,
  as described by the base repository. Confirm the actual worker versions.
- The worker requires a compatible environment's installed dependencies; it does
  not silently install packages, update drivers, or retry failures.

## Private inputs and preconditions

Provide a local runtime manifest, its exact checkpoint and reference TorchScript
policy, and the complete robot USD dependency tree. They must be delivered
privately, using a separately authorized mechanism. An arbitrary public download
or a URL/token in the launch command is not an acceptable substitute.

The manifest schema matches the existing exported runtime interface:
`checkpoint_sha256`, `policy_sha256`, 31 unique `joint_names`, 31-element
`default_q`, `default_dq`, `stiffness`, `damping`, `effort_limit`, and `env_cfg`.
The effective startup state and drive vectors must agree with it. The USD's own
SHA256 is supplied explicitly; referenced mesh/material files must also be
included and their bundle integrity checked by the private delivery mechanism.

Supply the exact deployed Git commit to `--expected-commit`. Launch from a clean,
independent evaluation branch, not a modified training tree. The worker verifies
HEAD; the controller must also verify that the delivered working tree is clean.

## Invocation

Run `supervisor.py` with the verified IsaacLab Python executable using `--python`
and a new receipt file using `--receipt`. After `--`, pass worker arguments:

- `--repository`: clean checkout of this branch
- `--expected-commit`: exact SHA of the deployed revision
- `--manifest`, `--checkpoint`, `--policy`: verified private local files
- `--usd`, `--usd-sha256`: verified local robot USD and expected identity
- `--output`: a new output directory, never overwrite a previous attempt
- optional `--seconds` (default 30, maximum 30) and `--seed` (default 42)

Test the CPU utilities using `python -m unittest discover -s <this-directory> -v`.
Provisioning a Flux draft, choosing billing limits, fetching private inputs,
running a paid task, and publishing results are **not** performed by these files.
Flux commands must use its documented `gm-run` entrypoint and verified image
interpreter. Never assume `python` points to the IsaacLab environment.

## Experimental controls and limitations

One environment; plane; initial yaw and horizontal velocity zero; command
`vx=1, vy=0, heading=0`; 20ms policy period; original manifest physics step;
observation noise and interval pushes disabled; no optimizer or training loop.
Startup mass/material randomization is **retained with a fixed seed**, and actual
values are exported. This is not proof of identical nominal dynamics to MuJoCo.
The rollout stops before a requested automatic reset can replace terminal state,
or when height falls below 0.5m / tilt exceeds 60 degrees.

The exact existing TorchScript is used without re-export; the checkpoint identity
is independently checked against the manifest, not loaded into a new policy.
Joint ordering and actual drive arrays must match. The currently audited interface
routes 20 of 31 joints; after the first step its effective target buffer must
match that routing, including zero targets on the remaining driven joints.
An unavailable or changed private buffer API is a failed compatibility check,
not permission to assume parity or silently switch controllers.

`trace.json` contains body-frame signed/absolute lateral velocity source samples,
world positions, height/tilt/yaw and commands. `physics.json` contains measured
masses/inertias/materials/armature when the image exposes the required getters;
unavailable getters are explicit. Summaries exclude samples before 2s, and omit
post-warmup means if termination happens earlier. `report.json` records input and
trace identities. These files remain private until a separately approved result
upload. Complete Flux task logs must still be retrieved and checked externally.

## Lifecycle and cost boundaries

The supervisor allows at most 300s initialization, 180s rollout/save phase, 60s
cleanup, and 600s total before terminating **its own child process group**, with
10s TERM/KILL waits. It never retries. A zero process exit alone is insufficient:
a bounded canonical JSON report and SHA256 plus successful cleanup are required.
Missing cleanup, report, or nonzero child exit is a failed attempt.

The supervisor default-denies inheritance of arbitrary environment variables,
retaining an explicit simulator/runtime allowlist, and does not mirror general
simulator stdout or exception messages. Only structured diagnostic events appear
in its output; account secrets must never be supplied to the worker at all.

**A worker watchdog is not a Flux task TTL or a billing cap.** A controller must
separately establish task/project authorization, resource currency and billing
unit, a total wall-clock/cost limit, terminal-state polling, platform stop/cleanup,
and private result retrieval before running anything. Persist task-create/run
intent and attempted flags before the corresponding request; an uncertain
response must be reconciled by reading the server, not by retrying a create/run.