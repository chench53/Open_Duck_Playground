import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from playground.common.onnx_infer import OnnxInfer
from playground.common.utils import LowPassActionFilter

from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase

USE_MOTOR_SPEED_LIMITS = True
GAIT_PHASE_PERIOD_STEPS = 50


class MjInfer(MJInferBase):
    def __init__(
        self,
        model_path: str,
        onnx_model_path: str,
        standing: bool,
    ):
        super().__init__(model_path)

        self.standing = standing
        self.head_control_mode = self.standing

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.COMMANDS_RANGE_X = [-0.15, 0.15]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

        self.NECK_PITCH_RANGE = [-0.34, 1.1]
        self.HEAD_PITCH_RANGE = [-0.78, 0.78]
        self.HEAD_YAW_RANGE = [-1.5, 1.5]
        self.HEAD_ROLL_RANGE = [-0.5, 0.5]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.mjx_manual_command = [0.0] * 7
        self.mjx_manual_viewer_active = False
        self.mjx_manual_reset_requested = False

        self.gait_phase_i = 0
        self.gait_phase = np.array([1.0, 0.0])
        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        contacts = self.get_feet_contacts(data)

        # if not self.standing:
        # ref = self.PRM.get_reference_motion(*command[:3], self.imitation_i)

        obs = np.concatenate(
            [
                gyro,
                accelerometer,
                # gravity,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                # ref if not self.standing else np.array([]),
                # [self.imitation_i]
                self.gait_phase,
            ]
        )

        return obs

    def key_callback(self, keycode):
        print(f"key: {keycode}")
        if keycode == 32:  # space: release all virtual joystick axes
            self.commands = [0.0] * len(self.commands)
            return
        if keycode == 72:  # h
            self.head_control_mode = not self.head_control_mode
        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0
        if not self.head_control_mode:
            if keycode == 265:  # arrow up
                lin_vel_x = self.COMMANDS_RANGE_X[1]
            if keycode == 264:  # arrow down
                lin_vel_x = self.COMMANDS_RANGE_X[0]
            if keycode == 263:  # arrow left
                lin_vel_y = self.COMMANDS_RANGE_Y[1]
            if keycode == 262:  # arrow right
                lin_vel_y = self.COMMANDS_RANGE_Y[0]
            if keycode == 81:  # q
                ang_vel = self.COMMANDS_RANGE_THETA[1]
            if keycode == 69:  # e
                ang_vel = self.COMMANDS_RANGE_THETA[0]
            if keycode == 80:  # p
                self.phase_frequency_factor += 0.1
            if keycode == 77:  # m
                self.phase_frequency_factor -= 0.1
        else:
            neck_pitch = 0
            head_pitch = 0
            head_yaw = 0
            head_roll = 0
            if keycode == 265:  # arrow up
                head_pitch = self.NECK_PITCH_RANGE[1]
            if keycode == 264:  # arrow down
                head_pitch = self.NECK_PITCH_RANGE[0]
            if keycode == 263:  # arrow left
                head_yaw = self.HEAD_YAW_RANGE[1]
            if keycode == 262:  # arrow right
                head_yaw = self.HEAD_YAW_RANGE[0]
            if keycode == 81:  # q
                head_roll = self.HEAD_ROLL_RANGE[1]
            if keycode == 69:  # e
                head_roll = self.HEAD_ROLL_RANGE[0]

            self.commands[3] = neck_pitch
            self.commands[4] = head_pitch
            self.commands[5] = head_yaw
            self.commands[6] = head_roll

        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel

    def mjx_manual_key_callback(self, keycode):
        if keycode == 32:
            self.mjx_manual_command = [0.0] * 7
            print("Manual command set to zero.", flush=True)
            return
        if keycode == 82 and self.mjx_manual_viewer_active:  # r
            self.mjx_manual_reset_requested = True
            print("Reset requested.", flush=True)
            return

        command = [0.0] * 7
        if keycode == 265:
            command[0] = self.COMMANDS_RANGE_X[1]
        elif keycode == 264:
            command[0] = self.COMMANDS_RANGE_X[0]
        elif keycode == 263:
            command[1] = self.COMMANDS_RANGE_Y[1]
        elif keycode == 262:
            command[1] = self.COMMANDS_RANGE_Y[0]
        elif keycode == 81:
            command[2] = self.COMMANDS_RANGE_THETA[1]
        elif keycode == 69:
            command[2] = self.COMMANDS_RANGE_THETA[0]
        else:
            return

        self.mjx_manual_command = command
        print(
            f"Manual command: vx={command[0]:.2f}, vy={command[1]:.2f}, "
            f"yaw={command[2]:.2f}",
            flush=True,
        )

    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback,
            ) as viewer:
                counter = 0
                while True:

                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)

                    counter += 1

                    if counter % self.decimation == 0:
                        if not self.standing:
                            self.gait_phase_i = (
                                self.gait_phase_i + self.phase_frequency_factor
                            ) % GAIT_PHASE_PERIOD_STEPS
                            phase = (
                                self.gait_phase_i / GAIT_PHASE_PERIOD_STEPS * 2 * np.pi
                            )
                            self.gait_phase = np.array(
                                [np.cos(phase), np.sin(phase)]
                            )
                        obs = self.get_obs(
                            self.data,
                            self.commands,
                        )
                        self.saved_obs.append(obs)
                        action = self.policy.infer(obs)

                        # self.action_filter.push(action)
                        # action = self.action_filter.get_filtered_action()

                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()

                        self.motor_targets = (
                            self.default_actuator + action * self.action_scale
                        )

                        if USE_MOTOR_SPEED_LIMITS:
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets
                                - self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                                self.prev_motor_targets
                                + self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                            )

                            self.prev_motor_targets = self.motor_targets.copy()

                        # head_targets = self.commands[3:]
                        # self.motor_targets[5:9] = head_targets
                        self.data.ctrl = self.motor_targets.copy()

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))

    def run_mjx_manual_viewer(self, checkpoint_path: str, seed: int = 0):
        import jax
        import jax.numpy as jp
        from brax.training import checkpoint
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
        from mujoco_playground.config import locomotion_params

        from playground.common.randomize import domain_randomize
        from playground.open_duck_mini_v2.joystick import Joystick

        env = Joystick(task="flat_terrain")
        ppo_config = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )
        networks = ppo_networks.make_ppo_networks(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
            **dict(ppo_config.network_factory),
        )
        params = checkpoint.load(checkpoint_path)
        policy = jax.jit(
            ppo_networks.make_inference_fn(networks)(params, deterministic=False)
        )
        rng = jax.random.PRNGKey(seed)
        rng, model_key, reset_key = jax.random.split(rng, 3)
        batched_model, model_axes = domain_randomize(
            env.mjx_model, jax.random.split(model_key, 1)
        )
        env._mjx_model = jax.tree_util.tree_map(
            lambda value, axis: value[0] if axis == 0 else value,
            batched_model,
            model_axes,
            is_leaf=lambda value: value is None,
        )
        reset_env = jax.jit(env.reset)
        step_env = jax.jit(env.step)
        state = reset_env(reset_key)
        self.mjx_manual_viewer_active = True
        self.mjx_manual_command = [0.0] * 7
        self.mjx_manual_reset_requested = False
        episode_steps = 0
        fallen = False

        def apply_manual_command(current_state):
            info = dict(current_state.info)
            command = jp.asarray(self.mjx_manual_command)
            info["command"] = command
            obs = dict(current_state.obs)
            obs["state"] = obs["state"].at[6:13].set(command)
            return current_state.replace(info=info, obs=obs)

        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.mjx_manual_key_callback,
            ) as viewer:
                print(
                    "Stochastic MJX policy ready. Arrows move, Q/E turn, "
                    "Space stops, R resets after a fall.",
                    flush=True,
                )
                while viewer.is_running():
                    step_start = time.time()
                    if self.mjx_manual_reset_requested:
                        rng, reset_key = jax.random.split(rng)
                        state = apply_manual_command(reset_env(reset_key))
                        episode_steps = 0
                        fallen = False
                        self.mjx_manual_reset_requested = False

                    if not fallen:
                        state = apply_manual_command(state)
                        rng, action_key = jax.random.split(rng)
                        action, _ = policy(
                            {"state": state.obs["state"][None, :]}, action_key
                        )
                        state = step_env(state, action[0])
                        episode_steps += 1
                        self.data.qpos[:] = np.asarray(
                            jax.device_get(state.data.qpos)
                        )
                        self.data.qvel[:] = np.asarray(
                            jax.device_get(state.data.qvel)
                        )
                        self.data.ctrl[:] = np.asarray(
                            jax.device_get(state.data.ctrl)
                        )
                        mujoco.mj_forward(self.model, self.data)
                        if bool(np.asarray(jax.device_get(state.done))):
                            print(
                                f"Fell after {episode_steps} steps; press R to reset.",
                                flush=True,
                            )
                            fallen = True

                    viewer.sync()
                    delay = env.dt - (time.time() - step_start)
                    if delay > 0:
                        time.sleep(delay)
        except KeyboardInterrupt:
            pass
        finally:
            self.mjx_manual_viewer_active = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    # parser.add_argument("-k", action="store_true", default=False)
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument("--standing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mjx-manual-viewer", action="store_true")

    parser.add_argument("--checkpoint-path", type=str)

    args = parser.parse_args()

    mjinfer = MjInfer(args.model_path, args.onnx_model_path, args.standing)
    if args.mjx_manual_viewer:
        if args.checkpoint_path is None:
            parser.error("--checkpoint-path is required with --mjx-manual-viewer")
        mjinfer.run_mjx_manual_viewer(args.checkpoint_path, seed=args.seed)
    else:
        mjinfer.run()
