# Workspace Guide

## Source of Truth

- The canonical source tree is `E:\cccodes\robot\Open_Duck_Playground`.
- Do not edit or train from a second source checkout under `/home/chench53/cccodes`.
- WSL imports the editable package from `/mnt/e/cccodes/robot/Open_Duck_Playground`. Verify with:
  `python -c "import playground; print(playground.__file__)"`
- A snapshot of the former WSL checkout is preserved at `/home/chench53/cccodes/Open_Duck_Playground_source_snapshot_20261006.tar.gz`.

## WSL Environment And Resources

- The WSL virtual environment is `/home/chench53/cccodes/Open_Duck_Playground_Resources/.venv`.
- JAX compilation cache is in `.tmp` beneath the runner's working directory.
- `--output_dir` selects where checkpoints, ONNX exports, and TensorBoard events are saved. The verified reproduction used workspace `checkpoints/reproduce_20261007`; older resource directories must also be preserved.
- `/home/chench53/cccodes/Open_Duck_Playground` is a compatibility symlink to the Resources directory. It is not a source checkout.
- Preserve `.venv`, `.tmp`, checkpoint directories, ONNX files, `models`, and recorded data when cleaning resources.

## Running From WSL

Use the WSL environment but point imports at the Windows workspace. Run from
the workspace so the imitation reference data's relative path resolves:

```bash
cd /mnt/e/cccodes/robot/Open_Duck_Playground
env PYTHONPATH=/mnt/e/cccodes/robot/Open_Duck_Playground \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
  /home/chench53/cccodes/Open_Duck_Playground_Resources/.venv/bin/python \
  -m playground.open_duck_mini_v2.runner --env joystick --task flat_terrain_backlash \
  --num_timesteps 300000000 --num_envs 2048 --batch_size 256 --num_minibatches 32 \
  --num_evals 16 --num_eval_envs 128 --num_resets_per_eval 1 --seed 0 \
  --output_dir checkpoints/reproduce_20261007
```

For warm-start training, pass an Orbax checkpoint directory to `--restore_checkpoint_path`; do not pass its `.onnx` export.
Preserve upstream rewards and the default enabled joystick imitation reward unless
the user explicitly requests a training change. The reference coefficients are required.

## Viewer Modes

- Normal ONNX inference uses the deterministic exported policy and `mujoco_infer.py --onnx_model_path <onnx>`.
- The initial input mode is walking. Arrow keys control translation, `Q/E` turn, `H` toggles head control, and `P`/semicolon adjust phase speed. Each key replaces the other velocity axes; release does not clear commands. There is no dedicated reset or stochastic MJX viewer in this version.
- Keep inference observations aligned with training. Do not reintroduce the former accelerometer X bias of 1.3: it prevented forward walking in same-policy A/B checks.

## Current Model Caveat

The local 302,284,800-step policy passed six corrected 15-second stand, forward,
and turn checks and 12-second keyboard tests. Forward yaw drift remains. Keep
claims tied to explicit evaluation results; these checks do not certify robustness
or real hardware. Training artifacts are ignored by Git and must not be committed.