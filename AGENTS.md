# Workspace Guide

## Source of Truth

- The canonical source tree is `E:\cccodes\robot\Open_Duck_Playground`.
- Do not edit or train from a second source checkout under `/home/chench53/cccodes`.
- WSL imports the editable package from `/mnt/e/cccodes/robot/Open_Duck_Playground`. Verify with:
  `python -c "import playground; print(playground.__file__)"`
- A snapshot of the former WSL checkout is preserved at `/home/chench53/cccodes/Open_Duck_Playground_source_snapshot_20261006.tar.gz`.

## WSL Environment And Resources

- The WSL virtual environment is `/home/chench53/cccodes/Open_Duck_Playground_Resources/.venv`.
- JAX compilation cache is under `/home/chench53/cccodes/Open_Duck_Playground_Resources/.tmp`.
- Checkpoints, exported ONNX files, and TensorBoard event files live in subdirectories of `/home/chench53/cccodes/Open_Duck_Playground_Resources`.
- `/home/chench53/cccodes/Open_Duck_Playground` is a compatibility symlink to the Resources directory. It is not a source checkout.
- Preserve `.venv`, `.tmp`, checkpoint directories, ONNX files, `models`, and recorded data when cleaning resources.

## Running From WSL

Use the WSL environment but point imports at the Windows workspace:

```bash
cd /home/chench53/cccodes/Open_Duck_Playground_Resources
env PYTHONPATH=/mnt/e/cccodes/robot/Open_Duck_Playground \
  .venv/bin/python /mnt/e/cccodes/robot/Open_Duck_Playground/playground/open_duck_mini_v2/runner.py \
  --env joystick --task flat_terrain \
  --num_timesteps 150000000 --num_envs 128 \
  --output_dir /home/chench53/cccodes/Open_Duck_Playground_Resources/checkpoints_new
```

For warm-start training, pass an Orbax checkpoint directory to `--restore_checkpoint_path`; do not pass its `.onnx` export.

## Viewer Modes

- Normal ONNX inference uses the deterministic exported policy and `mujoco_infer.py --onnx_model_path <onnx>`.
- Interactive stochastic MJX control uses `--mjx-manual-viewer --checkpoint-path <Orbax checkpoint>`. Pass the checkpoint's ONNX export as `--onnx_model_path` for CLI compatibility. Arrow keys control translation, `Q/E` turn, Space stops, and `R` resets after a fall.
- The ordinary ONNX Viewer and MJX checkpoint Viewer are different policy paths; do not treat their episode lengths as directly interchangeable.

## Current Model Caveat

The current locomotion policy has not demonstrated reliable walking. Keep claims about stability tied to an explicit Viewer rollout or evaluation result; the torso collision geom prevents falling through the floor but does not stabilize the policy.