# Open Duck Playground

# Installation 

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## WSL2 setup

The canonical source stays in the Windows workspace at
`E:\cccodes\robot\Open_Duck_Playground`. WSL imports it through `/mnt/e`; keep
the virtual environment, JAX cache, checkpoints, ONNX exports, and logs in
`/home/chench53/cccodes/Open_Duck_Playground_Resources`.

Install/update the editable package and dependencies in the WSL environment:

```bash
uv pip install --python /home/chench53/cccodes/Open_Duck_Playground_Resources/.venv/bin/python \
  --editable /mnt/e/cccodes/robot/Open_Duck_Playground
```

Verify the imported source and JAX device before training:

```bash
cd /home/chench53/cccodes/Open_Duck_Playground_Resources
PYTHONPATH=/mnt/e/cccodes/robot/Open_Duck_Playground \
  .venv/bin/python -c "import playground, jax; print(playground.__file__); print(jax.devices())"
```

# Training

The `open_duck_mini_v2` joystick environment uses PPO with MJX. The current
training setup does not use reference motions or an imitation reward. It learns
to track sampled velocity commands, with frequent zero-command samples for
standing, plus upright-orientation and fall-termination costs.

Run a full training job from WSL2:

```bash
cd /home/chench53/cccodes/Open_Duck_Playground_Resources
env PYTHONPATH=/mnt/e/cccodes/robot/Open_Duck_Playground \
  XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest \
  .venv/bin/python /mnt/e/cccodes/robot/Open_Duck_Playground/playground/open_duck_mini_v2/runner.py \
  --env joystick --task flat_terrain --num_timesteps 150000000 \
  --num_envs 128 --output_dir /home/chench53/cccodes/Open_Duck_Playground_Resources/checkpoints_locomotion
```

## Tensorboard

```bash
cd /home/chench53/cccodes/Open_Duck_Playground_Resources
uv tool run --python 3.12 --from tensorboard tensorboard \
  --logdir checkpoints_locomotion --host 0.0.0.0 --port 6007
```

# Inference 

Infer mujoco

(for now this is specific to open_duck_mini_v2)

```bash
cd /home/chench53/cccodes/Open_Duck_Playground_Resources
env PYTHONPATH=/mnt/e/cccodes/robot/Open_Duck_Playground \
  .venv/bin/python /mnt/e/cccodes/robot/Open_Duck_Playground/playground/open_duck_mini_v2/mujoco_infer.py \
  --onnx_model_path <path_to_.onnx>
```

With WSLg enabled, this opens the MuJoCo viewer. Arrow keys control forward,
backward, and lateral velocity; Q/E turn; Space clears the virtual joystick
commands; H toggles head-control input; P/M change gait phase speed. Commands
stay active until changed or cleared with Space.

For interactive stochastic MJX inference and fall reset, pass the matching
Orbax checkpoint directory with `--mjx-manual-viewer --checkpoint-path` (also
provide its ONNX export as `--onnx_model_path` for CLI compatibility). Arrow
keys control translation, Q/E turn, Space stops, and R resets after a fall. The
MJX checkpoint Viewer and deterministic ONNX Viewer are different policy paths;
their rollouts should not be compared as if they were the same evaluation.

# Documentation

## Project structure : 

```
.
├── pyproject.toml
├── README.md
├── playground
│   ├── common
│   │   ├── export_onnx.py
│   │   ├── onnx_infer.py
│   │   ├── poly_reference_motion.py
│   │   ├── randomize.py
│   │   ├── rewards.py
│   │   └── runner.py
│   ├── open_duck_mini_v2
│   │   ├── base.py
│   │   ├── data
│   │   │   └── polynomial_coefficients.pkl
│   │   ├── joystick.py
│   │   ├── mujoco_infer.py
│   │   ├── constants.py
│   │   ├── runner.py
│   │   └── xmls
│   │       ├── assets
│   │       ├── open_duck_mini_v2_no_head.xml
│   │       ├── open_duck_mini_v2.xml
│   │       ├── scene_mjx_flat_terrain.xml
│   │       ├── scene_mjx_rough_terrain.xml
│   │       └── scene.xml
```

## Adding a new robot

Create a new directory in `playground` named after `<your robot>`. You can copy the `open_duck_mini_v2` directory as a starting point.

You will need to:
- Edit `base.py`: Mainly renaming stuff to match you robot's name
- Edit `constants.py`: specify the names of some important geoms, sensors etc
  - In your `mjcf`, you'll probably have to add some sites, name some bodies/geoms and add the sensors. Look at how we did it for `open_duck_mini_v2`
- Add your `mjcf` assets in `xmls`. 
- Edit `joystick.py` : to choose the rewards you are interested in
  - Note: for now there is still some hard coded values etc. We'll improve things on the way
- Edit `runner.py`



# Notes

Inspired from https://github.com/kscalelabs/mujoco_playground


## Current win

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
```
