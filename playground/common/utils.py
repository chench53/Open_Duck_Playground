import jax.numpy as jp
import jax

try:
    from mujoco_playground._src.collision import geoms_colliding
except ModuleNotFoundError:
    def geoms_colliding(data, geom_id, other_geom_id):
        geom = data.contact.geom
        matches = ((geom[:, 0] == geom_id) & (geom[:, 1] == other_geom_id)) | (
            (geom[:, 0] == other_geom_id) & (geom[:, 1] == geom_id)
        )
        return jp.any(matches & (data.contact.dist < 0.0))


class LowPassActionFilter:
    def __init__(self, control_freq, cutoff_frequency=30.0):
        self.last_action = 0
        self.current_action = 0
        self.control_freq = float(control_freq)
        self.cutoff_frequency = float(cutoff_frequency)
        self.alpha = self.compute_alpha()

    def compute_alpha(self):
        return (1.0 / self.cutoff_frequency) / (
            1.0 / self.control_freq + 1.0 / self.cutoff_frequency
        )

    def push(self, action: jax.Array) -> None:
        self.current_action = jp.array(action)

    def get_filtered_action(self) -> jax.Array:
        self.last_action = (
            self.alpha * self.last_action + (1 - self.alpha) * self.current_action
        )
        return self.last_action