import pybullet as p
import numpy as np


class Robot:
    def __init__(self, simulation_env):
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
        self.current_target = None
        self.movement_active = False

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
        # Handle both location names and direct positions
        if isinstance(target, str):
            location = self.env.world.get_location(target)
            if location is None:
                print(f"Location '{target}' not found")
                return 'failure'
            target_pos = location.center
        else:
            target_pos = np.array(target)

        # Set up movement state
        self.current_target = np.array([target_pos[0], target_pos[1], 0])
        self.movement_active = True

        # Run simulation until we reach the target or timeout
        for _ in range(1000):
            if not self.movement_active:
                return 'success'
            self.env.step(1)

        # Timeout - stop movement
        self.movement_active = False
        p.resetBaseVelocity(
            self.base_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0]
        )
        return 'failure'

    def handle_move(self):
        assert self.movement_active and self.current_target is not None

        pos, ori = p.getBasePositionAndOrientation(self.base_id)
        pos = np.array(pos)

        # Calculate direction to target
        goal_vec = self.current_target - pos
        goal_vec[2] = 0  # Keep movement in XY plane
        dist = np.linalg.norm(goal_vec)

        # Check if we've reached the target
        if dist < 0.1:
            p.resetBaseVelocity(
                self.base_id,
                linearVelocity=[0, 0, 0],
                angularVelocity=[0, 0, 0]
            )
            self.movement_active = False
            self.current_target = None
        else:
            # Move towards target
            v = 3 * goal_vec / dist
            p.resetBaseVelocity(
                self.base_id,
                linearVelocity=v,
                angularVelocity=[0, 0, 0]
            )

    def on_pre_step(self):
        if self.movement_active and self.current_target is not None:
            self.handle_move()

    def on_post_step(self):
        # Handle any post-physics updates
        # e.g., update state, check collisions
        pass
