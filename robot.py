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
        self.action_target = None
        self.activity = set()

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
        self.action_target = np.array([target_pos[0], target_pos[1], 0])
        self.activity.add("move")

        # Run simulation until we reach the target or timeout
        for _ in range(1000):
            if not "move" in self.activity:
                return 'success'
            self.env.step(1)

        # Timeout - stop movement
        self.activity.remove("move")
        self.action_target = None
        p.resetBaseVelocity(
            self.base_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0]
        )

    def handle_move(self):
        assert "move" in self.activity and self.action_target is not None
        # Need to assert that action_target is a numpy array or list of length 3

        pos, ori = p.getBasePositionAndOrientation(self.base_id)
        pos = np.array(pos)

        # Calculate direction to target
        goal_vec = self.action_target - pos
        goal_vec[2] = 0  # Keep movement in XY plane
        dist = np.linalg.norm(goal_vec)

        # Check if we've reached the target
        if dist < 0.1:
            p.resetBaseVelocity(
                self.base_id,
                linearVelocity=[0, 0, 0],
                angularVelocity=[0, 0, 0]
            )
            self.activity.remove("move")
            self.action_target = None
        else:
            # Move towards target
            v = 3 * goal_vec / dist
            p.resetBaseVelocity(
                self.base_id,
                linearVelocity=v,
                angularVelocity=[0, 0, 0]
            )

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
                return 'failure'
        else:
            target_id = target_name_or_id

        # Set up grab state
        self.action_target = target_id
        self.activity.add("grab")

        # Run simulation until we grab the object or timeout
        for _ in range(1000):
            if not "grab" in self.activity:
                return 'success'
            self.env.step(1)

        # Timeout - failed to grab
        self.activity.remove("grab")
        self.action_target = None

        return 'failure'

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
        if dist < 0.25:
            offset = np.array([0, 0, 0.25])
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
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            pos = np.array(pos)
            target_pos = np.array(ee_pos) + offset
            diff = target_pos - pos

            joint_angles = p.calculateInverseKinematics(self.robot_id, 6, pos+0.5*diff+3*offset) # Fix moving up
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

    def on_pre_step(self):
        if "move" in self.activity and self.action_target is not None:
            self.handle_move()
        if "grab" in self.activity and self.action_target is not None:
            self.handle_grab()

    def on_post_step(self):
        # Handle any post-physics updates
        # e.g., update state, check collisions
        pass
