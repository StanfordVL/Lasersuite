import os
import mujoco_py
import numpy as np
import pybullet as pb
import xml.etree.ElementTree as ET
from collections import OrderedDict

from ..utils.transform_utils import convert_quat, mat2quat
from ..robots import SingleArm
from ..models import assets_root
from ..models.arenas import TableArena
from ..models.objects import BoxObject, DoorObject
from ..models.tasks import TableTopTask, UniformRandomSampler, DoorTask
from ..controllers import get_pybullet_server, load_controller_config, controller_factory
from .robot_env import RobotEnv


class DoorOpen(RobotEnv):
	"""
	This class corresponds to the lifting task for a single robot arm.
	"""

	def __init__(
			self,
			robots,
			controller_configs=None,
			gripper_types="default",
			gripper_visualizations=False,
			# initialization_noise=0.02,
			initialization_noise="default",
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
			control_freq=25,
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
		super().__init__(robots=robots, controller_configs=controller_configs, gripper_types=gripper_types,
						 gripper_visualizations=gripper_visualizations, initialization_noise=initialization_noise,
						 use_camera_obs=use_camera_obs, use_indicator_object=use_indicator_object,
						 has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer,
						 render_camera=render_camera, render_collision_mesh=render_collision_mesh,
						 render_visual_mesh=render_visual_mesh, control_freq=control_freq, horizon=horizon,
						 ignore_done=ignore_done, camera_names=camera_names, camera_heights=camera_heights,
						 camera_widths=camera_widths, camera_depths=camera_depths)

	def _load_model(self):
		"""
		Loads an xml model, puts it in self.model
		"""
		super()._load_model()
		assert isinstance(self.robots[0],
						  SingleArm), "Error: Expected one single-armed robot! Got {} type instead.".format(
			type(self.robots[0]))

		self.dist_threshold = 0.01
		self.excess_force_penalty_mul = 0.05
		self.excess_torque_penalty_mul = 0.5
		self.torque_threshold_max = 100 * 0.1
		self.pressure_threshold_max = 100
		self.energy_penalty = 0
		self.ee_accel_penalty = 0
		self.action_delta_penalty = 0
		self.handle_reward = True
		self.arm_collision_penalty = -1

		self.handle_final_reward = 1
		self.handle_shaped_reward = 0.5
		self.max_hinge_diff = 0.05
		self.max_hinge_vel = 0.1
		self.final_reward = 10
		self.door_shaped_reward = 15
		self.hinge_goal = 1.04
		self.velocity_penalty = 1
		self.change_door_friction = False
		self.door_damping_max = 50
		self.door_damping_min = 50
		self.door_friction_max = 10
		self.door_friction_min = 5
		self.gripper_on_handle = True
		self.use_door_state = True
		self.table_origin = [0.50 + self.table_full_size[0] / 2, 0, 0]

		self.robots[0].robot_model.set_base_xpos([0, 0, 0])
		self.mujoco_arena = TableArena(table_full_size=self.table_full_size)
		self.mujoco_arena.set_origin(self.table_origin)
		self.door = DoorObject()
		self.mujoco_objects = OrderedDict([("Door", self.door)])
		if self.gripper_on_handle:
			self._init_qpos = np.array(
				[-0.01068642, -0.05599809, 0.22389938, -1.81999415, -1.54907898, 2.72220116, 2.28768505])

		self.model = DoorTask(self.mujoco_arena, [robot.robot_model for robot in self.robots], self.mujoco_objects)
		if self.change_door_friction:
			damping = np.random.uniform(high=np.array([self.door_damping_max]), low=np.array([self.door_damping_min]))
			friction = np.random.uniform(high=np.array([self.door_friction_max]),
										 low=np.array([self.door_friction_min]))
			self.model.set_door_damping(damping)
			self.model.set_door_friction(friction)

		self.model.place_objects()

	def _reset_internal(self, xoffset=-1, **kwargs):
		"""
		Resets simulation internal configurations.
		"""
		super()._reset_internal()
		# inherited class should reset positions of objects
		self.model.place_objects()
		# reset joint positions
		self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes] = np.array(self._init_qpos)
		self.sim.data.qpos[self.robots[0]._ref_gripper_joint_pos_indexes] = np.array([0.001, -0.001])
		self.sim.data.qpos[self.sim.model.joint_names.index("door_hinge")] = 0
		self.timestep = 0
		self.wiped_sensors = []
		self.touched_handle = 0
		self.collisions = 0
		self.f_excess = 0
		self.t_excess = 0
		self.joint_torques = 0
		self.ee_acc = np.zeros(6)
		self.ee_force = np.zeros(3)

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
		pr = self.robots[0].robot_model.naming_prefix
		handle_id = self.sim.model.site_name2id("door_handle")
		self.handle_position = self.sim.data.site_xpos[handle_id]
		handle_orientation_mat = self.sim.data.site_xmat[handle_id].reshape(3, 3)
		handle_orientation = mat2quat(handle_orientation_mat)
		hinge_id = self.sim.model.get_joint_qpos_addr("door_hinge")

		self.hinge_qpos = np.array((self.sim.data.qpos[hinge_id])).reshape(-1, )
		self.hinge_qvel = np.array((self.sim.data.qvel[hinge_id])).reshape(-1, )
		# low-level object information
		if self.use_object_obs:
			# path = [self.get_body_pos(name) for name in self.path_names]
			contact = self._check_gripper_contact()
			di['task_state'] = np.array([[0, 1][contact]])
			if self.use_door_state:
				di['task_state'] = np.concatenate([di['task_state'], self.hinge_qpos, self.hinge_qvel])
			di["target_pos"] = [0, 0, 0]
			di["object-state"] = np.concatenate([di["task_state"], di["robot0_eef_pos"]])
		return di

	def _check_gripper_contact(self):
		"""
		Returns True if gripper is in contact with an object.
		"""
		collision = False
		for contact in self.sim.data.contact[:self.sim.data.ncon]:
			if self.sim.model.geom_id2name(contact.geom1) in self.robots[
				0].gripper.contact_geoms or self.sim.model.geom_id2name(contact.geom2) in self.robots[
				0].gripper.contact_geoms:
				collision = True
				break
		return collision

	def _check_arm_contact(self):
		"""
		Returns True if the arm is in contact with another object.
		"""
		collision = False
		for contact in self.sim.data.contact[:self.sim.data.ncon]:
			if self.sim.model.geom_id2name(contact.geom1) in self.robots[
				0].robot_model.contact_geoms or self.sim.model.geom_id2name(contact.geom2) in self.robots[
				0].robot_model.contact_geoms:
				collision = True
				break
		return collision

	def _check_q_limits(self, debug=False):
		"""
		Returns True if the arm is in joint limits or very close to.
		"""
		joint_limits = False
		tolerance = 0.1
		for (idx, (q, q_limits)) in enumerate(
				zip(self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes], self.sim.model.jnt_range)):
			if not (q > q_limits[0] + tolerance and q < q_limits[1] - tolerance):
				if debug: print("Joint limit reached in joint " + str(idx))
				joint_limits = True
				# self.robots[0].joint_limit_count += 1
		return joint_limits

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
		reward = 0
		pg = self.robots[0].gripper.naming_prefix
		grip_id = self.sim.model.site_name2id(pg + "grip_site")
		eef_position = self.sim.data.site_xpos[grip_id]

		force_sensor_id = self.sim.model.sensor_name2id(pg + "force_ee")
		self.force_ee = self.sim.data.sensordata[force_sensor_id * 3: force_sensor_id * 3 + 3]
		total_force_ee = np.linalg.norm(np.array(self.ee_force))

		torque_sensor_id = self.sim.model.sensor_name2id(pg + "torque_ee")
		self.torque_ee = self.sim.data.sensordata[torque_sensor_id * 3: torque_sensor_id * 3 + 3]
		total_torque_ee = np.linalg.norm(np.array(self.torque_ee))

		self.hinge_diff = np.abs(self.hinge_goal - self.hinge_qpos)

		# Neg Reward from collisions of the arm with the table
		if self._check_arm_contact() or self._check_q_limits():
			reward += self.arm_collision_penalty
		else:
			# add reward for touching handle or being close to it
			if self.handle_reward:
				dist = np.linalg.norm(eef_position[0:2] - self.handle_position[0:2])
				if dist < self.dist_threshold and abs(eef_position[2] - self.handle_position[2]) < 0.02:
					self.touched_handle = 1
					reward += self.handle_reward
				else:
					# if robot starts 0.3 away and dist_threshold is 0.05: [0.005, 0.55] without scaling
					reward += (self.handle_shaped_reward * (1 - np.tanh(3 * dist))).squeeze()
					self.touched_handle = 0
			# penalize excess force
			if total_force_ee > self.pressure_threshold_max:
				reward -= self.excess_force_penalty_mul * total_force_ee
				self.f_excess += 1
			# penalize excess torque
			if total_torque_ee > self.torque_threshold_max:
				reward -= self.excess_torque_penalty_mul * total_torque_ee
				self.t_excess += 1
			# award bonus either for opening door or for making process toward it
			if self.hinge_diff < self.max_hinge_diff and abs(self.hinge_qvel) < self.max_hinge_vel:
				reward += self.final_reward
			else:
				reward += (self.door_shaped_reward * (np.abs(self.hinge_goal) - self.hinge_diff)).squeeze()
				reward -= (self.hinge_qvel * self.velocity_penalty).squeeze()
		# penalize for jerkiness
		reward -= self.energy_penalty * np.sum(np.abs(self.joint_torques))
		reward -= self.ee_accel_penalty * np.mean(abs(self.ee_acc))
		return reward * self.reward_scale / 2.25

	def done(self, debug=False):
		terminated = False
		if self._check_q_limits():
			if debug: print(40 * '-' + " JOINT LIMIT " + 40 * '-')
			terminated = True

		# Prematurely terminate if contacting the table with the arm
		if self._check_arm_contact():
			if debug: print(40 * '-' + " COLLIDED " + 40 * '-')
			terminated = True
		return terminated

	def _visualization(self):
		"""
		Do any needed visualization here. Overrides superclass implementations.
		"""
		# color the gripper site appropriately based on distance to cube
		if self.robots[0].gripper_visualization:
			dist = np.sum(np.square(self.get_body_pos("target") - self.sim.data.get_site_xpos(
				self.robots[0].gripper.visualization_sites["grip_site"])))
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

	'''
	def get_viewer(self, mode):
		if self.viewer is None:
			self.viewer = mujoco_py.MjViewer(self.sim) if mode in ["human"] else mujoco_py.MjRenderContextOffscreen(
				self.sim, -1) if mode in ["rgb_array", "depth_array"] else None
			self.viewer.vopt.geomgroup[0] = (1 if self.render_collision_mesh else 0)
			self.viewer.vopt.geomgroup[1] = (1 if self.render_visual_mesh else 0)
			self.viewer._hide_overlay = True
			self.viewer._render_every_frame = True
			self.viewer.cam.trackbodyid = 0
			self.viewer.cam.azimuth = 270
			self.viewer.cam.elevation = -90
			self.viewer.cam.distance = 1
		return self.viewer
	'''
