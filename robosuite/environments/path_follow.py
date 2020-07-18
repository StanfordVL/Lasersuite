import os
import numpy as np
import pybullet as pb
import xml.etree.ElementTree as ET
from collections import OrderedDict

from ..utils.transform_utils import convert_quat
from ..robots import SingleArm
from ..models import assets_root
from ..models.arenas import TableArena
from ..models.objects import BoxObject
from ..models.tasks import TableTopTask, UniformRandomSampler
from ..controllers import get_pybullet_server, load_controller_config, controller_factory
from ..controllers.ee_ik import PybulletServer
from .robot_env import RobotEnv


class PathFollow(RobotEnv):
    """
    This class corresponds to the lifting task for a single robot arm.
    """
    def __init__(
        self,
        robots,
        controller_configs=None,
        gripper_types="default",
        gripper_visualizations=False,
        initialization_noise=0.02,
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=2.25,
        reward_shaping=False,
        placement_initializer=None,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=100,
        horizon=250,
        ignore_done=False,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        height=0
        ):
        """
        Args:
            robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms) Note: Must be a single single-arm robot!
            controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a custom controller. Else, uses the default controller for this specific task. Should either be single dict if same controller is to be used for all robots or else it should be a list of the same length as "robots" param
            gripper_types (str or list of str): type of gripper, used to instantiate gripper models from gripper factory. Default is "default", which is the default grippers(s) associated with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model overrides the default gripper. Should either be single str if same gripper type is to be used for all robots or else it should be a list of the same length as "robots" param
            gripper_visualizations (bool or list of bool): True if using gripper visualization. Useful for teleoperation. Should either be single bool if gripper visualization is to be used for all robots or else it should be a list of the same length as "robots" param
            initialization_noise (float or list of floats): The scale factor of uni-variate Gaussian random noise applied to each of a robot's given initial joint positions. Setting this value to "None" or 0.0 results in no noise being applied. Should either be single float if same noise value is to be used for all robots or else it should be a list of the same length as "robots" param
            table_full_size (3-tuple): x, y, and z dimensions of the table.
            table_friction (3-tuple): the three mujoco friction parameters for the table.
            use_camera_obs (bool): if True, every observation includes rendered image(s)
            use_object_obs (bool): if True, include object (cube) information in the observation.
            reward_scale (float): Scales the normalized reward function by the amount specified
            reward_shaping (bool): if True, use dense rewards.
            placement_initializer (ObjectPositionSampler instance): if provided, will be used to place objects on every reset, else a UniformRandomSampler is used by default.
            use_indicator_object (bool): if True, sets up an indicator object that is useful for debugging.
            has_renderer (bool): If true, render the simulation state in a viewer instead of headless mode.
            has_offscreen_renderer (bool): True if using off-screen rendering
            render_camera (str): Name of camera to render if `has_renderer` is True.
            render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.
            render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.
            control_freq (float): how many control signals to receive in every second. This sets the amount of simulation time that passes between every action input.
            horizon (int): Every episode lasts for exactly @horizon timesteps.
            ignore_done (bool): True if never terminating the environment (ignore @horizon).
            camera_names (str or list of str): name of camera to be rendered. Should either be single str if same name is to be used for all cameras' rendering or else it should be a list of cameras to render.
                Note: At least one camera must be specified if @use_camera_obs is True.
                Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each robot's camera list).
            camera_heights (int or list of int): height of camera frame. Should either be single int if same height is to be used for all cameras' frames or else it should be a list of the same length as "camera names" param.
            camera_widths (int or list of int): width of camera frame. Should either be single int if same width is to be used for all cameras' frames or else it should be a list of the same length as "camera names" param.
            camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single bool if same depth setting is to be used for all cameras or else it should be a list of the same length as "camera names" param.
        """
        # First, verify that only one robot is being inputted
        self.region_height = height
        self._check_robot_configuration(robots)
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs
        # self.placement_initializer = UniformRandomSampler( x_range=[-0.03, 0.03], y_range=[-0.03, 0.03], ensure_object_boundary_in_range=False, z_rotation=None)
        super().__init__(robots=robots, controller_configs=controller_configs, gripper_types=gripper_types, gripper_visualizations=gripper_visualizations, initialization_noise=initialization_noise, use_camera_obs=use_camera_obs, use_indicator_object=use_indicator_object, has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer, render_camera=render_camera, render_collision_mesh=render_collision_mesh, render_visual_mesh=render_visual_mesh, control_freq=control_freq, horizon=horizon, ignore_done=ignore_done, camera_names=camera_names, camera_heights=camera_heights, camera_widths=camera_widths, camera_depths=camera_depths)

    def reward(self, action=None):
        """
        Reward function for the task.

        The dense un-normalized reward has three components.

            Reaching: in [0, 1], to encourage the arm to reach the cube
            Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): unused for this task

        Returns:
            reward (float): the reward
        """
        ef_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        target_pos = self.get_body_pos("target")
        path = [self.get_body_pos(name) for name in self.path_names]
        target_dist = ef_pos-target_pos
        path_dists = [ef_pos-path_pos for path_pos in path]
        reward_goal = -np.linalg.norm(target_dist)*2
        reward_path = -np.min(np.linalg.norm(path_dists, axis=-1))
        reward = reward_goal + reward_path
        return reward * self.reward_scale / 2.25

    def done(self):
        state = self._get_observation()
        ef_pos = state["robot0_eef_pos"]
        target_pos = state["target_pos"]
        return np.linalg.norm(ef_pos-target_pos)<0.01

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        # Verify the correct robot has been loaded
        assert isinstance(self.robots[0], SingleArm), "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))
        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        # load model for table top workspace
        self.mujoco_arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction)
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()
        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])
        
        root = self.robots[0].robot_model.root
        option = root.find("option")
        option.set("gravity", "0 0 0")
        worldbody = root.find("worldbody")
        self.path_names = []
        path = ET.Element("body")
        path.set("name", "path")
        path.set("pos", "0 0 0")
        for i in range(10):
            name = f"path{i}"
            point = ET.Element("body")
            point.set("name", name)
            point.set("pos", "0 0 0")
            point.append(ET.fromstring("<geom conaffinity='0' group='1' contype='0' pos='0 0 0' rgba='0.8 0.2 0.4 0.8' size='.005' type='sphere'/>"))
            path.append(point)
            self.path_names.append(name)
        worldbody.append(path)
        self.range = 1.0
        self.origin = np.array([-0, 0, 1.0])
        self.size = np.array([0.2, 0.2+self.region_height, self.region_height])
        self.space = "box"
        size = self.size[0] if self.space == "sphere" else np.maximum(self.size, 0.001)
        size_str = lambda x: ' '.join([f'{p}' for p in x])
        space = f"<body name='space' pos='{size_str(self.origin)}'><geom conaffinity='0' group='1' contype='0' name='space' rgba='0.9 0.9 0.9 0.4' size='{size_str(size)}' type='{self.space}'/></body>"
        worldbody.append(ET.fromstring(space))
        target = "<body name='target' pos='0 -0.20 .2'><geom conaffinity='0' group='1' contype='0' name='target' pos='0 0 0' rgba='0.4 0.8 0.2 1' size='.01' type='sphere'/></body>"
        worldbody.append(ET.fromstring(target))

        self.model = TableTopTask(self.mujoco_arena, [robot.robot_model for robot in self.robots])
        self.model.place_objects()

    def _reset_internal(self, xoffset=-1, **kwargs):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        origin = self.origin + xoffset*np.array([0, 0.2, 0])
        target_pos = origin + self.range*self.size*np.random.uniform(-1, 1, size=self.size.shape)
        ef_to_pos = origin + self.range*self.size*np.random.uniform(-1, 1, size=self.size.shape)
        offset = np.array([0,0,0.04]) * ((origin+self.size)[0]-ef_to_pos[0])/(2*self.size[0])
        qpos = np.copy(self.init_qpos)
        qpos[self.robots[0].inverse_controller.joint_index] = self.robots[0].inverse_controller.inverse_kinematics(ef_to_pos+offset, None)
        qvel = self.init_qvel + np.random.uniform(low=-.005, high=.005, size=self.sim.model.nv)
        self.sim.model.body_pos[self.sim.model.body_names.index("target")] = target_pos
        self.sim.model.body_pos[self.sim.model.body_names.index("space")] = origin
        self.set_state(qpos=qpos, qvel=qvel)
        target_pos = self.get_body_pos("target")
        ef_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        points = np.linspace(ef_pos, target_pos, len(self.path_names))
        path_indices = [self.sim.model.body_names.index(name) for name in self.path_names]
        for i,point in zip(path_indices, points):
            self.sim.model.body_pos[i] = point
        
    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
        # low-level object information
        if self.use_object_obs:
            path = [self.get_body_pos(name) for name in self.path_names]
            di["target_pos"] = path[-1]
            di["task_state"] = np.concatenate([*path, di["robot0_eef_pos"]])
        return di

    def _visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """
        # color the gripper site appropriately based on distance to cube
        if self.robots[0].gripper_visualization:
            dist = np.sum(np.square(self.get_body_pos("target") - self.sim.data.get_site_xpos(self.robots[0].gripper.visualization_sites["grip_site"])))
            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5
            self.sim.model.site_rgba[self.robots[0].eef_site_id] = rgba

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable
        """
        if type(robots) is list:
            assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"

