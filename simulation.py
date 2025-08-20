import dotenv
import os
import pybullet as p
import pybullet_data
import time

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from typing import TYPE_CHECKING

from world import World
from robot import Robot

dotenv.load_dotenv()

if TYPE_CHECKING:
	from world import World


class TimeStepController:
	def __init__(self, dt=0.01, real_time=True):
		self.dt = dt
		self.real_time = real_time
		self.last_step_time = time.time()

	def step(self):
		if self.real_time:
			current_time = time.time()
			sleep_time = self.dt - (current_time - self.last_step_time)
			if sleep_time > 0:
				time.sleep(sleep_time)
			self.last_step_time = time.time()
		else:
			# Run as fast as possible
			time.sleep(self.dt)

class SimulationEnvironment:
	def __init__(self, world: 'World'):
		# Initialize PyBullet
		self.physics_client = p.connect(p.GUI)
		p.setGravity(0, 0, -9.8)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())

		# Initialize timing and simulation components
		self.time_controller = TimeStepController()
		self.subscribers = []

		# Place the camera above the origin, looking straight down
		p.resetDebugVisualizerCamera(
			cameraDistance=5.0,         # How high above the scene (zoom level)
			cameraYaw=0,                # Yaw doesn't matter much when looking straight down
			cameraPitch=-45,#-89.999,       # Almost -90° for top-down (can't be exactly -90°)
			cameraTargetPosition=[0, 0, 0]  # Look at the center of the world
		)

		# Load ground plane
		p.loadURDF("plane.urdf")

		# Set up world
		self.world = world

	def add_subscriber(self, subscriber):
		"""Add a component that needs to be notified of simulation steps."""
		self.subscribers.append(subscriber)

	def remove_subscriber(self, subscriber):
		"""Remove a component from the simulation updates."""
		if subscriber in self.subscribers:
			self.subscribers.remove(subscriber)

	def _notify_subscribers(self, event_type):
		"""Notify all subscribers of a simulation event."""
		for subscriber in self.subscribers:
			if hasattr(subscriber, f'on_{event_type}'):
				getattr(subscriber, f'on_{event_type}')()

	def step(self, num_steps=1):
		"""Advance the simulation by the specified number of steps."""
		for _ in range(num_steps):
			# Pre-physics updates
			self._notify_subscribers('pre_step')

			# Physics step
			p.stepSimulation()

			# Post-physics updates
			self._notify_subscribers('post_step')

			# Control timing
			self.time_controller.step()

	def disconnect(self):
		"""Clean up and disconnect from PyBullet."""
		p.disconnect(self.physics_client)


if __name__ == "__main__":
	# Create a simulation environment
	world = World.create_default_world()
	sim = SimulationEnvironment(world)
	sim.world.create_default_physical_objects()

	# Add a subscriber (e.g., a robot)
	if os.environ.get("OPENAI_API_KEY", None):
		model = ModelFactory.create(
		  model_platform=ModelPlatformType.OPENAI,
		  model_type="gpt-4.1-mini",
		  model_config_dict={"temperature": 0.0},
		)
	else:
		model = None

	robot = Robot(sim, model)
	sim.add_subscriber(robot)

	# Run the simulation
	sim.step(200)

	# Give the robot commands
	robot.grab("cube")

	for way_point in sim.world.get_path_between("Front Door", "Infront of Kitchen Shelf"):
		print(f"Moving to {way_point}")
		print(robot.move_to(way_point))


	# Place the cube at the shelf
	robot.place("Bottom Kitchen Shelf")
	sim.step(100)
	robot.grab("cube")
	sim.step(100)
	robot.place("Top Kitchen Shelf")

	
	robot.percieve()
	
	# Back to door
	for way_point in sim.world.get_path_between("Infront of Kitchen Shelf", "Front Door"):
		print(f"Moving to {way_point}")
		print(robot.move_to(way_point))

	# Delay before invoking the robot's agent
	sim.step(1000)

	# Invoke the robot's agent
	robot.invoke("Please pick up the cube, move to the door and place the cube")

	# Step the simulation again
	while True:
		sim.step(1000)

	# Disconnect from PyBullet
	sim.disconnect()
