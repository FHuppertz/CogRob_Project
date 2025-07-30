import pybullet as p
import pybullet_data
import time
import numpy as np

from typing import Optional

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

        self.constraint_id = None
        self.held_object_id = None

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

        # Test Cube
        half_size = [0.1,0.1,0.1]
        cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_size)
        cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_size)
        self.cube_id = p.createMultiBody(baseMass=0.01, baseCollisionShapeIndex=cube_collision,
                                         baseVisualShapeIndex=cube_visual, basePosition=[1.0,0.0,0.1])

        # Add cube to objects dictionary
        global OBJECTS
        OBJECTS["cube"] = self.cube_id


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

    def Grab_X(self, target_name_or_id):
        """
        Grab an object by name or ID.

        Args:
            target_name_or_id: Either a string name of an object or a direct object ID

        Returns:
            str: 'success' if grabbing was successful, 'failure' otherwise
        """
        # Handle both object names and direct IDs
        if isinstance(target_name_or_id, str):
            target_id = OBJECTS.get(target_name_or_id.lower())
            if target_id is None:
                print(f"Object '{target_name_or_id}' not found in OBJECTS dictionary")
                return 'failure'
        else:
            target_id = target_name_or_id

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
                self.constraint_id = p.createConstraint(
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

                p.changeConstraint(self.constraint_id, maxForce=500, erp=1.0)

                # TODO: Make Gripper move up (Better)
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
                self.held_object_id = target_id
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

    def Place_Object(self, target_name_or_position):
        """
        Place the currently grabbed object at a location or position.

        Args:
            target_name_or_position: Either a string name of a location or [x, y, z] coordinates

        Returns:
            str: 'success' if placement was successful, 'failure' otherwise
        """
        # Handle both location names and direct positions
        if isinstance(target_name_or_position, str):
            location = find_location_by_name(target_name_or_position)
            if location is None:
                print(f"Location '{target_name_or_position}' not found")
                return 'failure'
            target_position = location.place_position
        else:
            target_position = target_name_or_position

        # Check if we're actually holding something
        if self.constraint_id is None:
            return 'failure'

        # Position above the target to ensure safe placement
        placement_pos = np.array(target_position)
        placement_pos[2] += 0.2  # Place slightly above the target

        # Move arm to placement position
        for _ in range(1000):
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

            if dist < 0.1:
                # Remove the constraint to release the object
                p.removeConstraint(self.constraint_id)
                self.constraint_id = None

                # Teleport the object to the exact target position
                p.resetBasePositionAndOrientation(self.held_object_id, target_position, [0,0,0,1])

                # Move arm slightly up to ensure release
                lift_pos = placement_pos.copy()
                lift_pos[2] += 0.1

                # Move to lift position
                lift_joint_angles = p.calculateInverseKinematics(self.robot_id, 6, lift_pos)
                for i, angle in enumerate(lift_joint_angles):
                    p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
                                            targetPosition=angle,
                                            force=500,
                                            positionGain=0.03,
                                            velocityGain=1.0,
                                            maxVelocity=2.0)

                self.Simulate(100)  # Brief simulation to settle
                return 'success'

            self.Simulate(1)

        return 'failure'



### World representation
class Location():
    def __init__(self, name, center, place_position=None):
        self.name = name
        self.center = np.array(center)
        # Default place position is slightly offset from center
        self.place_position = np.array(place_position) if place_position is not None else np.array([center[0], center[1], 0.5])
        self.neighbours = []

    def Next_To(self, neighbours):
        for neighbour in neighbours:
            if neighbour not in self.neighbours:
                self.neighbours.append(neighbour)
                neighbour.Next_To([self])

## Locations:
Door = Location("Door", [0,0], [0.0, 0.0, 0.5])
LivingRoom = Location("LivingRoom", [3,0], [3.0, 0.0, 0.5])
Fridge = Location("Fridge", [1,-2], [1.0, -2.0, 0.5])
Stove = Location("Stove", [3,-2], [3.0, -2.0, 0.5])
TV = Location("TV", [1,2], [1.0, 2.5, 0.5])

# Object dictionary - maps object names to their IDs in the simulation
OBJECTS = {}

# World representation - dictionary of all locations by name (lowercase keys)
WORLD = {
    "door": Door,
    "livingroom": LivingRoom,
    "fridge": Fridge,
    "stove": Stove,
    "tv": TV
}

# Location Relations:
Door.Next_To([LivingRoom, Fridge, TV])
LivingRoom.Next_To([TV, Door, Fridge, Stove])
Fridge.Next_To([Door, LivingRoom, Stove])
Stove.Next_To([Fridge, LivingRoom])
TV.Next_To([Door, LivingRoom])

# Path Finder
def find_location_by_name(name: str) -> Optional[Location]:
    """
    Find a location in the world by its name (case-insensitive).

    Args:
        name (str): The name of the location to find

    Returns:
        Location: The location object with the matching name, or None if not found
    """
    # Convert name to lowercase for consistent lookup
    return WORLD.get(name.lower())

class Node():
    def __init__(self, location):
        self.location = location
        self.f = 0
        self.g = 0
        self.path = []

def Path_From_To(start_name, end_name):
    start = find_location_by_name(start_name)
    end = find_location_by_name(end_name)

    if start is None or end is None:
        return []

    goal = False
    checked = [start]
    frontier = []

    current_Node = Node(start)
    frontier.append(current_Node)

    while frontier != [] and not goal:
        frontier.sort(key=lambda x: x.f)
        current_Node = frontier.pop(0)

        for neigh in current_Node.location.neighbours:
            if neigh not in checked:
                # Goal check
                if neigh == end:
                    goal = True
                    current_Node.path.append(end.center)
                    return current_Node.path
                checked.append(neigh)

                # Add new location to frontier and calcualte its f value
                next_Node = Node(neigh)
                next_Node.path = current_Node.path.copy()
                next_Node.path.append(neigh.center)
                next_Node.g = current_Node.g + np.linalg.norm(next_Node.location.center - current_Node.location.center)
                next_Node.f = next_Node.g + np.linalg.norm(end.center - neigh.center)

                frontier.append(next_Node)

    return []


print(Path_From_To("door", "stove"))

# TODO:
# Motion of Base (A* missing)
# Actual grabbing (kinda done)


#'''
Sim = Simulation()
print(Sim.Grab_X("cube"))
for way_point in Path_From_To("door", "stove"):
    print(Sim.Move_To(way_point))

for way_point in Path_From_To("stove", "tv"):
    print(Sim.Move_To(way_point))

print(Sim.Place_Object("tv"))

Sim.Simulate(10000)
p.disconnect()
#'''
