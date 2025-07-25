import pybullet as p
import pybullet_data
import time
import numpy as np

# Connect to simulation
p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load ground
p.loadURDF("plane.urdf")

# Create mobile base (cube)
base_size = [0.4, 0.4, 0.2]
base_start_pos = [0.0, 0.0, base_size[2] / 2]
base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in base_size])
base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in base_size])
base_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=base_collision,
                            baseVisualShapeIndex=base_visual, basePosition=base_start_pos)

# Load robot arm on top of base
robot_height = base_size[2]  # cube height
robot_id = p.loadURDF("kuka_iiwa/model.urdf", base_start_pos, useFixedBase=True)

# Define target positions (world frame)
target_positions = [[1.0, 0.0, 0.5], [0.5, 0.5, 0.5]]

def move_base_toward(current_pos, target_pos, step_size=0.01):
    """Move base gradually toward target."""
    diff = np.array(target_pos) - np.array(current_pos)
    dist = np.linalg.norm(diff)
    if dist < step_size:
        return target_pos, True  # Close enough
    direction = diff / dist
    new_pos = np.array(current_pos) + direction * step_size
    return new_pos.tolist(), False

for target in target_positions:
    print(f"Moving toward: {target}")
    target_xy = target[:2]
    end_eff_z = target[2]

    done = False
    while not done:
        # Get current base position
        base_pos, _ = p.getBasePositionAndOrientation(base_id)

        # Compute new base position
        new_xy, done = move_base_toward(base_pos[:2], target_xy)

        # Update base and robot position
        new_base_pos = [*new_xy, base_size[2] / 2]
        p.resetBasePositionAndOrientation(base_id, new_base_pos, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(robot_id, new_base_pos, [0, 0, 0, 1])

        # Compute relative target for arm IK
        relative_target = [
            target[0] - new_base_pos[0],
            target[1] - new_base_pos[1],
            target[2] - new_base_pos[2]
        ]
        joint_angles = p.calculateInverseKinematics(robot_id, 6, relative_target)

        # Apply joint positions
        for i, angle in enumerate(joint_angles):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=angle)

        # Step simulation
        p.stepSimulation()
        time.sleep(0.01)

# Disconnect
p.disconnect()

