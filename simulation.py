import pybullet as p
import pybullet_data
import time

from world import World
from robot import Robot


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
    def __init__(self, world):
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

    # Add a subscriber (e.g., a robot)
    robot = Robot(sim)
    sim.add_subscriber(robot)

    # Run the simulation
    sim.step(100)

    # Give robot a move command
    robot.move_to("tv")

    # Step the simulation again
    sim.step(100)

    # Disconnect from PyBullet
    sim.disconnect()
