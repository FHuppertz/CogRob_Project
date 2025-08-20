import pybullet as p
import numpy as np

from camel.agents import ChatAgent
from typing import TYPE_CHECKING

from toolkit import RobotToolkit

if TYPE_CHECKING:
	from simulation import SimulationEnvironment


class Robot:
	def __init__(self, simulation_env: SimulationEnvironment, model=None):
		self.env = simulation_env
		self.env.add_subscriber(self)

		# Initialize robot components
		# Create mobile base (cube)
		base_size = [0.7, 0.7, 0.3]
		base_start_pos = [0.0, 0.0, base_size[2] / 2]
		base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in base_size])
		base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in base_size])
		self.base_id = p.createMultiBody(baseMass=1000.0, baseCollisionShapeIndex=base_collision,
									baseVisualShapeIndex=base_visual, basePosition=base_start_pos)

		# Load robot arm on top of base
		# Adjust start position to be on top of the cube base
		robot_start_pos = [base_start_pos[0], base_start_pos[1], base_size[2]]
		self.robot_id = p.loadURDF("kuka_iiwa/model_2.urdf", robot_start_pos, useFixedBase=False)

		p.resetBasePositionAndOrientation(self.robot_id, robot_start_pos,[0.0,0.0,0.0,1.0])
		# Attach arm to base using fixed constraint
		p.createConstraint(
			parentBodyUniqueId=self.base_id,
			parentLinkIndex=-1,
			childBodyUniqueId=self.robot_id,
			childLinkIndex=-1,
			jointType=p.JOINT_FIXED,
			jointAxis=[0, 0, 0],
			parentFramePosition=robot_start_pos,
			childFramePosition=base_start_pos
		)

		self.constraint_id = None
		self.held_object_id = None

		# Movement state
		self.action_target = None
		self.activity = set()

		# Path planning and following
		self.path = []
		self.waypoint_index = 0
		self._reset_path()

		# Agent
		system_message = "You are a robot assistant. Your goal is to help the user with their tasks.\n"
		"You have the following tools available to you to assist with tasks:\n"
		" - move_to: Move the robot base to a target location by name (str)\n"
		" - grab: Pick up an object by name (str)\n"
		" - place: Place the held object at the given location by name (str)\n"

		if model:
			self.toolkit = RobotToolkit(self)
			self.chat_agent = ChatAgent(
				system_message=system_message,
				model=model,
				tools=self.toolkit.get_tools(),
			)
		else:
			self.toolkit = None
			self.chat_agent = None

	@property
	def position(self):
		pos, ori = p.getBasePositionAndOrientation(self.base_id)
		return pos

	def invoke(self, task_prompt: str):
		if self.chat_agent:
			prompt = self.create_environment_prompt() + "\n"
			response = self.chat_agent.step(prompt + task_prompt)
			print(f"Agent response:\n{response.msgs[0].content}")
		else:
			print("No chat agent available, please provide a model")

	def create_environment_prompt(self) -> str:
		"""
		Create a prompt describing the environment of the robot.
		
		Returns:
			str: Formatted description of the environment
		"""
		environment_description = ""
		environment_description += self.env.world.get_locations_description()
		return environment_description

	def move_to(self, target):
		"""
		Move the robot base to a target position. Initiates movement through
		setting the target position and activating the movement state. Movement
		is handled by the on_pre_step, which is called when env.step() is called.

		Args:
			target: Either a string name of a location or [x, y] coordinates

		Returns:
			str: 'success' if movement was successful, 'failure' otherwise
		"""
		current_location_name = self.env.world.get_current_location(self.position)
		
		# Handle both location names and direct positions
		if isinstance(target, str):
			# Generate path to target location
			if current_location_name is None:
				print("Current location not found")
				return {
					'status': 'failure',
					'message': "Current location not found"
				}
			
			self.path = self.env.world.get_path_between(current_location_name, target)
			if not self.path:
				# Fallback to direct movement if no path found
				location = self.env.world.get_location(target)
				if location is None:
					print(f"Location '{target}' not found")
					return {
						'status': 'failure',
						'message': f"Location '{target}' not found"
					}
				target_pos = location.center
			else:
				# Use path planning
				self.waypoint_index = 0
				target_pos = np.array(self.path[0])
		else:
			# Direct coordinates - no path planning
			target_pos = np.array(target)

		# Set up movement state
		self.action_target = np.array([target_pos[0], target_pos[1], 0])
		self.activity.add("move")

		# Run simulation until we reach the target or timeout
		for _ in range(1000):
			if not "move" in self.activity:
				return {
					'status': 'success',
					'message': f'You successfully reached the target {target}'
				}
			self.env.step(1)

		# Timeout - stop movement
		self.activity.remove("move")
		self.action_target = None

		# Reset path irrespective of whether we were following one
		self._reset_path()

		p.resetBaseVelocity(
			self.base_id,
			linearVelocity=[0, 0, 0],
			angularVelocity=[0, 0, 0]
		)

		return {
			'status': 'timeout',
			'message': f'Timed out while moving to target {target}'
		}

	def handle_move(self):
		assert "move" in self.activity and self.action_target is not None
		# Need to assert that action_target is a numpy array or list of length 3

		pos = np.array(self.position)

		# Calculate direction to target
		goal_vec = self.action_target - pos
		goal_vec[2] = 0  # Keep movement in XY plane
		dist = np.linalg.norm(goal_vec)

		# Check if we've reached the target
		if dist < 0.1:
			# Check if we're following a path
			if self.path and self.waypoint_index < len(self.path) - 1:
				# Move to next waypoint
				self.waypoint_index += 1
				next_waypoint = self.path[self.waypoint_index]
				self.action_target = np.array([next_waypoint[0], next_waypoint[1], 0])
			else:
				# Reached final target
				p.resetBaseVelocity(
					self.base_id,
					linearVelocity=[0, 0, 0],
					angularVelocity=[0, 0, 0]
				)
				self.activity.remove("move")
				self.action_target = None
				self._reset_path()
		else:
			# Move towards target
			v = 3 * goal_vec / dist
			p.resetBaseVelocity(
				self.base_id,
				linearVelocity=v,
				angularVelocity=[0, 0, 0]
			)

	def _reset_path(self):
		self.path = []
		self.waypoint_index = 0

	def grab(self, target_name_or_id):
		"""
		Grab an object by name or ID. Initiates grabbing through setting the
		target object and activating the grab state. Grabbing is handled by
		on_pre_step, which is called when env.step() is called.

		Args:
			target_name_or_id: Either a string name of an object or a direct object ID

		Returns:
			str: 'success' if grabbing was successful, 'failure' otherwise
		"""
		# Handle both object names and direct IDs
		if isinstance(target_name_or_id, str):
			target_id = self.env.world.get_object(target_name_or_id)
			if target_id is None:
				print(f"Object '{target_name_or_id}' not found in world objects")
				return {
					'status': 'failure',
					'message': f'Object "{target_name_or_id}" not found'
				}
		else:
			target_id = target_name_or_id

		# Set up grab state
		self.action_target = target_id
		self.activity.add("grab")

		# Run simulation until we grab the object or timeout
		for _ in range(1000):
			if not "grab" in self.activity:
				return {
					'status': 'success',
					'message': 'Object grabbed successfully'
				}
			self.env.step(1)

		# Timeout - failed to grab
		self.activity.remove("grab")
		self.action_target = None
		p.changeConstraint(self.constraint_id, maxForce=500, erp=1.0)

		# Move Gripper up
		pos = np.array(self.position)

		joint_angles = p.calculateInverseKinematics(self.robot_id, 6, pos+np.array([0., 0., 1.5])) # Fix moving up
		for i, angle in enumerate(joint_angles):
			p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
									targetPosition=angle,
									force=500,
									positionGain=0.03,
									velocityGain=1.0,
									maxVelocity=2.0
			)
		return {
			'status': 'timeout',
			'message': 'Timed out trying to grab object'
		}

	def handle_grab(self):
		assert "grab" in self.activity and self.action_target is not None
		assert type(self.action_target) is int

		target_pos, target_ori = p.getBasePositionAndOrientation(self.action_target)
		joint_angles = p.calculateInverseKinematics(self.robot_id, 6, target_pos)

		# Get end-effector position
		ee_index = 6
		# Get gripper pose
		ee_pos, ee_ori = p.getLinkState(self.robot_id, ee_index)[4:6]

		dist = np.linalg.norm(np.array(target_pos) - np.array(ee_pos))
		if dist < 0.5:
			offset = np.array([0.25, 0, 0])
			# Constraints to simulate grabbing (modified ChatGPT)
			p.resetBasePositionAndOrientation(
				self.action_target,
				np.array(ee_pos)+offset,
				target_ori
			)

			# Transform from parent world frame to parent local frame
			parent_inv_pos, parent_inv_ori = p.invertTransform(ee_pos, ee_ori)

			# Compute child pose relative to parent
			rel_pos, rel_ori = p.multiplyTransforms(
				parent_inv_pos, parent_inv_ori,
				np.array(ee_pos)+offset, target_ori
			)
			# Create constraint as grab
			self.constraint_id = p.createConstraint(
				parentBodyUniqueId=self.robot_id,
				parentLinkIndex=6,
				childBodyUniqueId=self.action_target,
				childLinkIndex=-1,
				jointType=p.JOINT_FIXED,
				jointAxis=[0, 0, 0],
				parentFramePosition=rel_pos,
				childFramePosition=[0, 0, 0],
				parentFrameOrientation=rel_ori,
				childFrameOrientation=[0, 0, 0, 1]
			)

			p.changeConstraint(self.constraint_id, maxForce=500, erp=1.0)

			# Move Gripper up
			pos = np.array(self.position)

			joint_angles = p.calculateInverseKinematics(self.robot_id, 6, pos+np.array([0., 0., 1.5])) # Fix moving up
			for i, angle in enumerate(joint_angles):
				p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
										targetPosition=angle,
										force=500,
										positionGain=0.03,
										velocityGain=1.0,
										maxVelocity=2.0
				)

			self.held_object_id = self.action_target
			self.activity.remove("grab")
			self.action_target = None
		else:
			# Move arm towards target
			for i, angle in enumerate(joint_angles):
				p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
										targetPosition=angle,
										force=500,
										positionGain=0.03,
										velocityGain=1.0,
										maxVelocity=2.0
				)

	def place(self, target_name_or_position):
		"""
		Place the currently grabbed object at a location or position. Initiates
		placement through setting the target position and activating the place state.
		Placement is handled by on_pre_step, which is called when env.step() is called.

		Args:
			target_name_or_position: Either a string name of a location or [x, y, z] coordinates

		Returns:
			str: 'success' if placement was successful, 'failure' otherwise
		"""
		# Handle both location names and direct positions
		if isinstance(target_name_or_position, str):
			location = self.env.world.get_location(target_name_or_position)
			if location is None:
				print(f"Location '{target_name_or_position}' not found")
				return {
					"status": "failure",
					"message": f"Location '{target_name_or_position}' not found"
				}
			target_position = location.place_position
		else:
			target_position = target_name_or_position

		# Check if we're actually holding something
		if self.constraint_id is None:
			return {
				"status": "failure",
				"message": "You are not holding anything"
			}

		# Set up place state
		self.action_target = np.array(target_position)
		self.activity.add("place")

		# Run simulation until we place the object or timeout
		for _ in range(1000):
			if not "place" in self.activity:
				return {
					"status": "success",
					"message": "Object placed successfully"
				}
			self.env.step(1)

		# Timeout - failed to place
		self.activity.remove("place")
		self.action_target = None

		# Move Gripper up
		pos = np.array(self.position)

		joint_angles = p.calculateInverseKinematics(self.robot_id, 6, pos+np.array([0., 0., 1.5])) # Fix moving up
		for i, angle in enumerate(joint_angles):
			p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
									targetPosition=angle,
									force=500,
									positionGain=0.03,
									velocityGain=1.0,
									maxVelocity=2.0
			)
		return {
			"status": "failure",
			"message": "Failed to place object"
		}

	def handle_place(self):
		assert "place" in self.activity and self.action_target is not None
		assert isinstance(self.action_target, np.ndarray) and len(self.action_target) == 3

		placement_pos = self.action_target.copy()

		# Move arm to placement position
		joint_angles = p.calculateInverseKinematics(self.robot_id, 6, placement_pos)

		# Apply joint angles
		for i, angle in enumerate(joint_angles):
			p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
									targetPosition=angle,
									force=500,
									positionGain=0.03,
									velocityGain=1.0,
									maxVelocity=2.0)

		# Check if we're close enough to the target position
		ee_pos, _ = p.getLinkState(self.robot_id, 6)[4:6]
		dist = np.linalg.norm(np.array(placement_pos) - np.array(ee_pos))

		if dist < 0.5:
			# Remove the constraint to release the object
			p.removeConstraint(self.constraint_id)
			self.constraint_id = None

			# Teleport the object to the exact target position
			p.resetBasePositionAndOrientation(self.held_object_id, self.action_target, [0,0,0,1])

			self.held_object_id = None
			self.activity.remove("place")
			self.action_target = None
			
			# Move Gripper up
			pos = np.array(self.position)

			joint_angles = p.calculateInverseKinematics(self.robot_id, 6, pos+np.array([0., 0., 1.5])) # Fix moving up
			for i, angle in enumerate(joint_angles):
				p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
										targetPosition=angle,
										force=500,
										positionGain=0.03,
										velocityGain=1.0,
										maxVelocity=2.0
				)

	def on_pre_step(self):
		if "move" in self.activity and self.action_target is not None:
			self.handle_move()
		if "grab" in self.activity and self.action_target is not None:
			self.handle_grab()
		if "place" in self.activity and self.action_target is not None:
			self.handle_place()

	def on_post_step(self):
		# Handle any post-physics updates
		# e.g., update state, check collisions
		pass
