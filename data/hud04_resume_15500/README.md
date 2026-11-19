# HU_D04 step-15500 continuation input: ordinary Git parts

Publication of the robot assets and complete checkpoint in this public repository
was explicitly authorized by the user on 2026-11-19. No account keys, private
configuration, signed URLs or task payloads are included. This notice does not
grant additional rights beyond the original assets' and checkpoint's terms.

The two non-LFS parts reconstruct exactly the original 69,716,909-byte archive:

`0078b6fdfd7f8a9fb083614e371f027134b20fde8ee9bcaa7271a052d0a8ce00`

It contains 118 HU_D04 robot asset files and the original full `model_15500.pt`
checkpoint, SHA-256:

`f3d91b4b0e536c9151d5d27c6a6adc30e44e968010547f3ffec359f16af856c7`

The checkpoint is trusted research input from the originating user's run. Do not
deserialize unknown replacement checkpoints. CPU input verification never loads
pickle or imports the simulator.

## CPU preflight (repository root)

```bash
python tools/hud04_speed_resume/test_bootstrap.py
python tools/hud04_speed_resume/verify_git_inputs.py --output-dir outputs/git-input-preflight
```

Choose a new output directory if that directory already exists; preflight refuses
to overwrite it. The worker verifies exact part order, basenames, per-part sizes
and hashes, rejects links and LFS pointers, streams into a new archive and checks
the whole-archive identity before the existing safe extraction path.

## Training integration

`train_resume.py --git-inputs` selects this repository-local manifest. The existing
private output-configuration interface is still required (`--private-config-file`);
no real private path or capability is included here. This revision does not claim
that private configuration delivery or result upload is already available.

Git delivery does not require personal-storage mounting or external checkpoint
downloads. It moves transfer to platform clone; clone time can still count towards
billing. Use an independently recorded full commit, not a moving branch alone.

The training recipe is unchanged: hu_d04_run, 4096 environments, command-x
1.2–2.0 m/s, at most 2000 additional updates and the existing 3600-second worker
bound. No actuator-limit changes, no automatic retry, and no GPU is launched by
the scripts above. CPU checks cannot prove resumed optimizer updates or speed gains.