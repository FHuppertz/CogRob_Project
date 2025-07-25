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

        # Load ground
        p.loadURDF("plane.urdf")

        # Create mobile base (cube)
        base_size = [0.7, 0.7, 0.3]
        base_start_pos = [0.0, 0.0, base_size[2] / 2]
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in base_size])
        base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in base_size])
        self.base_id = p.createMultiBody(baseMass=10.0, baseCollisionShapeIndex=base_collision,
                                baseVisualShapeIndex=base_visual, basePosition=base_start_pos)

        # Load robot arm on top of base
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", base_start_pos, useFixedBase=True)
        p.setCollisionFilterPair(self.robot_id, self.base_id, -1, -1, 0)


        # Test Cube
        half_size = [0.1,0.1,0.1]
        cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_size)
        cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_size)
        self.cube_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=cube_collision,
                                         baseVisualShapeIndex=cube_visual, basePosition=[1.0,0.0,0.1])

    def Move_To(self, target):
        # Attach Arm to base
        pos, ori = p.getBasePositionAndOrientation(self.base_id)
        p.resetBasePositionAndOrientation(self.robot_id, np.array(pos)+np.array([0.0,0.0,0.3]), ori) # QuickFix

    def Grab_X(self, target_id):
        target_pos, target_ori = p.getBasePositionAndOrientation(target_id)
        joint_angles = p.calculateInverseKinematics(self.robot_id, 6, target_pos)

        # Apply joint positions
        for i, angle in enumerate(joint_angles):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=angle)

    def Simulate(self, steps):
        for _ in range(steps):
            self.Move_To(0)
            self.Grab_X(self.cube_id)
            p.stepSimulation()
            time.sleep(0.01)


## TODO!!!!
# Motion of Base
# Fix QuickFix orientation
# Actual grabbing



Sim = Simulation()
Sim.Simulate(100000)
p.disconnect()
