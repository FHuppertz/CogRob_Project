import pybullet as p
import pybullet_data
import time
import numpy as np

class Simulation():
    def __init__(self):
        # Connect to simulation
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Place the camera above the origin, looking straight down
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,         # How high above the scene (zoom level)
            cameraYaw=0,                # Yaw doesn't matter much when looking straight down
            cameraPitch=-45,#-89.999,       # Almost -90° for top-down (can't be exactly -90°)
            cameraTargetPosition=[0, 0, 0]  # Look at the center of the world
        )

        # Load ground
        p.loadURDF("plane.urdf")

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
        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.base_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.robot_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=robot_start_pos,
            childFramePosition=base_start_pos
        )

        # Test Cube
        half_size = [0.1,0.1,0.1]
        cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_size)
        cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_size)
        self.cube_id = p.createMultiBody(baseMass=0.01, baseCollisionShapeIndex=cube_collision,
                                         baseVisualShapeIndex=cube_visual, basePosition=[1.0,0.0,0.1])


    def Move_To(self, target):
        for _ in range(1000): 
            target_vec = np.zeros(3)
            target_vec[0] = target[0]
            target_vec[1] = target[1]

            pos, ori = p.getBasePositionAndOrientation(self.base_id)
            pos = np.array(pos)
            ori = np.array(ori)

            goal_vec = target_vec - pos
            goal_vec[2] = 0
            dist = np.linalg.norm(goal_vec)

            if dist < 0.1:
                p.resetBaseVelocity(
                    self.base_id, 
                    linearVelocity=[0,0,0], 
                    angularVelocity=[0, 0, 0]
                ) 
                return 'success'
            
            v = 3 * goal_vec/dist
            p.resetBaseVelocity(
                self.base_id, 
                linearVelocity=v, 
                angularVelocity=[0, 0, 0]
            )

            pos, ori = p.getBasePositionAndOrientation(self.base_id)
            pos = np.array(pos)
            ori = np.array(ori)

            goal_vec = target_vec - pos
            goal_vec[2] = 0
            dist = np.linalg.norm(goal_vec)

            self.Simulate(1)


        p.resetBaseVelocity(
            self.base_id, 
            linearVelocity=[0,0,0], 
            angularVelocity=[0, 0, 0]
        ) 
        return 'failure'



    def Grab_X(self, target_id):
        for _ in range(1000):
            target_pos, target_ori = p.getBasePositionAndOrientation(target_id)
            joint_angles = p.calculateInverseKinematics(self.robot_id, 6, target_pos)

            # Get end-effector position
            ee_index = 6
            # Get gripper pose
            ee_pos, ee_ori = p.getLinkState(self.robot_id, ee_index)[4:6]

            dist = np.linalg.norm(np.array(target_pos) - np.array(ee_pos))
            if dist < 0.25:
                offset = np.array([0, 0, 0.25])
                # Contraints to simulate grabbing (modified ChatGPT)
                p.resetBasePositionAndOrientation(
                    target_id,
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
                # Create contraint as grab
                constraint_id = p.createConstraint(
                    parentBodyUniqueId=self.robot_id,
                    parentLinkIndex=6,
                    childBodyUniqueId=target_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=rel_pos,
                    childFramePosition=[0, 0, 0],
                    parentFrameOrientation=rel_ori,
                    childFrameOrientation=[0, 0, 0, 1]
                )

                p.changeConstraint(constraint_id, maxForce=500, erp=1.0)

                # TODO: Make Gripper move up (Better)
                pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                pos = np.array(pos)
                target_pos = np.array(ee_pos) + offset
                diff = target_pos - pos

                joint_angles = p.calculateInverseKinematics(self.robot_id, 6, pos+0.5*diff+3*offset)
                for i, angle in enumerate(joint_angles):
                    p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, 
                                            targetPosition=angle,
                                            force=500,
                                            positionGain=0.03,
                                            velocityGain=1.0,
                                            maxVelocity=2.0
                    )
                return 'success'

            for i, angle in enumerate(joint_angles):
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, 
                                        targetPosition=angle,
                                        force=500,
                                        positionGain=0.03,
                                        velocityGain=1.0,
                                        maxVelocity=2.0
                )

            self.Simulate(1)

        return 'failure'


    def Simulate(self, steps):
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(0.01)


## TODO!!!!
# Motion of Base
# Fix QuickFix orientation
# Actual grabbing



Sim = Simulation()
print(Sim.Move_To([1,-1]))
print(Sim.Grab_X(Sim.cube_id))
Sim.Simulate(10000)
p.disconnect()
