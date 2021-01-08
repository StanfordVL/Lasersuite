import numpy as np
from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.models.objects import BallObject


class TargetArena(TableArena):
    """
    Workspace that contains a tabletop with a target position.

    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
        coverage_factor (float): Fraction of table that will be sampled for dirt placement
    """

    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        coverage_factor=0.9
    ):
        self.coverage_factor = coverage_factor

        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset
        )

    def configure_location(self):
        """Configures correct locations for this arena"""
        # Run superclass first
        super().configure_location()

        # Grab reference to the table body in the xml
        table_subtree = self.worldbody.find(".//body[@name='{}']".format("table"))

        goal = self.sample_goal()
        target = BallObject(
                name="target",
                size=[0.01],
                rgba=[0, 1, 0, 1],
            )
        
        self.merge_asset(target)
        visual_c = target.get_visual(site=True)
        visual_c.set("pos", array_to_string([goal[0], goal[1], 2.75*self.table_half_size[2]]))
        #visual_c.find("site").set("pos", [0, 0, 0.005])
        #visual_c.find("site").set("rgba", array_to_string([0, 0, 0, 0]))
        table_subtree.append(visual_c)


    def reset_arena(self, sim):
        """Reset target location. Requires @sim (MjSim) reference to be passed in so that
        the Mujoco sim can be directly modified

        Args:
            sim (MjSim): Simulation instance containing this arena and tactile sensors
        """
        goal = self.sample_goal()
        position = np.array([goal[0], goal[1], 2.75*self.table_half_size[2]])
        body_id = sim.model.body_name2id('target')
        geom_id = sim.model.geom_name2id('target')
        sim.model.body_pos[body_id] = position
        sim.model.geom_rgba[geom_id] = [0, 1, 0, 1]


    def sample_goal(self):
        """
        Helper function to return sampled start position of a new target location

        Returns:
            np.array: the (x,y) value of the newly sampled target starting location
        """

        # First define the random direction that we will start at
        self.direction = np.random.uniform(-np.pi, np.pi)

        return np.array(
            (
                np.random.uniform(
                        -self.table_half_size[0] * self.coverage_factor + 0.01,
                        self.table_half_size[0] * self.coverage_factor - 0.01),
                np.random.uniform(
                        -self.table_half_size[1] * self.coverage_factor + 0.01,
                        self.table_half_size[1] * self.coverage_factor - 0.01)
            )
        )
