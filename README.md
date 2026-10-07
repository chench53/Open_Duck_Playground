# Open Duck Playground

# Installation 

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## WSL2 setup

The canonical source stays in the Windows workspace at
`E:\cccodes\robot\Open_Duck_Playground`. WSL imports it through `/mnt/e`. The
virtual environment is in `/home/chench53/cccodes/Open_Duck_Playground_Resources`.
The runner stores its JAX cache in `.tmp` beneath the working directory;
`--output_dir` selects the checkpoint, ONNX, and TensorBoard output directory.
Training artifacts and caches are ignored by Git.

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
setup preserves the upstream rewards and enables imitation by default through
`USE_IMITATION_REWARD=True`. It requires
`playground/open_duck_mini_v2/data/polynomial_coefficients.pkl`, generated with
[the reference motion generator](https://github.com/apirrone/Open_Duck_reference_motion_generator).
The standing environment disables imitation by default.

The following configuration was exercised on a WSL2 RTX 5070 Ti with 16 GB
VRAM and 15 GiB WSL RAM. It retains the upstream PPO network and reward setup:

```bash
cd /mnt/e/cccodes/robot/Open_Duck_Playground
mkdir -p checkpoints/reproduce_20261007
set -o pipefail
env PYTHONPATH=/mnt/e/cccodes/robot/Open_Duck_Playground \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 TF_NUM_INTRAOP_THREADS=4 \
  TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=4 MUJOCO_GL=egl \
  /home/chench53/cccodes/Open_Duck_Playground_Resources/.venv/bin/python -u \
  -m playground.open_duck_mini_v2.runner \
  --env joystick --task flat_terrain_backlash --num_timesteps 300000000 \
  --num_envs 2048 --batch_size 256 --num_minibatches 32 --num_evals 16 \
  --num_eval_envs 128 --num_resets_per_eval 1 --seed 0 \
  --output_dir checkpoints/reproduce_20261007 \
  2>&1 | tee checkpoints/reproduce_20261007/stdout.log
```

Resource parameters can be overridden with the flags above; omitted flags use
the upstream PPO defaults. Resume with `--restore_checkpoint_path` pointing to
an Orbax checkpoint directory, not an ONNX file. TensorFlow uses CPU for export
so it does not compete with JAX for GPU memory.

## Tensorboard

```bash
uv pip install --python /home/chench53/cccodes/Open_Duck_Playground_Resources/.venv/bin/python tensorboard
env CUDA_VISIBLE_DEVICES=-1 \
  /home/chench53/cccodes/Open_Duck_Playground_Resources/.venv/bin/tensorboard \
  --logdir /mnt/e/cccodes/robot/Open_Duck_Playground/checkpoints \
  --host 0.0.0.0 --port 6006
```

Open `http://localhost:6006/`. Both training and evaluation metrics are recorded.

# Inference 

Infer mujoco

(for now this is specific to open_duck_mini_v2)

```bash
cd /mnt/e/cccodes/robot/Open_Duck_Playground
env PYTHONPATH=/mnt/e/cccodes/robot/Open_Duck_Playground MUJOCO_GL=glfw \
  /home/chench53/cccodes/Open_Duck_Playground_Resources/.venv/bin/python \
  -m playground.open_duck_mini_v2.mujoco_infer \
  --onnx_model_path <path_to_.onnx> \
  --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml
```

With WSLg enabled, this opens the MuJoCo viewer. Arrow keys control forward,
backward, and lateral velocity; Q/E turn; H toggles head-control input; P and
semicolon change gait phase speed. The initial mode is walking and mode changes
are printed. Commands remain active after key release. Each key replaces the
other velocity axes, so simultaneous forward/turn control is not supported.
Press H to stop and switch modes, then H again to return to walking. This
version does not provide a stochastic MJX viewer or an R reset command.

## Reproduction results (2026-10-07)

- WSL Python 3.12.15, MuJoCo/MJX 3.14.0, JAX 0.6.2 with CUDA12,
  Playground 0.2.0, and Brax 0.14.2.
- Training completed 302,284,800 steps with exit code 0; checkpoints and ONNX
  exports were generated throughout training.
- The viewer formerly added 1.3 to accelerometer X, unlike the effective JAX
  training observation. Removing this inference-only bias improved the same
  policy's 15-second forward distance from 0.056 m to 1.091 m.
- Corrected inference passed six 15-second stand, forward, and forward/turn
  rollouts across the flat models with and without backlash. Keyboard forward
  control traveled about 1.61 m over 12 seconds without falling.
- Forward yaw drift remains (about 0.75 rad in the 15-second backlash rollout).
  These are short deterministic checks, not a guarantee of robust or hardware
  walking. Model weights, detailed reports, videos, and logs are local artifacts
  and are not included in this repository.

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
