import numpy as np
import pybullet as p

from typing import Dict, List, Optional


class Location():
    def __init__(self,
                 name: str,
                 center: List[float],
                 place_position: Optional[List[float]] = None
            ):
        self.name = name
        self.center = np.array(center)
        # Default place position is slightly offset from center
        self.place_position = np.array(place_position) if place_position is not None else np.array([center[0], center[1], 0.5])
        self.neighbours = []

    def next_to(self, neighbours: List['Location']) -> None:
        for neighbour in neighbours:
            if neighbour not in self.neighbours:
                self.neighbours.append(neighbour)
                neighbour.next_to([self])


class PlannerNode():
    def __init__(self, location: 'Location'):
        self.location = location
        self.f = 0
        self.g = 0
        self.path = []


class World():
    def __init__(self):
        self.locations: Dict[str, Location] = {}
        self.objects: Dict[str, int] = {}

    def add_location(self, location: Location) -> None:
        self.locations[location.name.lower()] = location

    def add_next_to(self, location: str, neighbours: List[str]) -> None:
        self.locations[location.lower()].next_to([self.locations[neighbour.lower()] for neighbour in neighbours])
        for neighbour in neighbours:
            neighbour_location = self.locations[neighbour.lower()]
            neighbour_location.next_to([self.locations[location.lower()]])

    def get_location(self, name: str) -> Optional[Location]:
        return self.locations.get(name.lower())

    def add_object(self, object_id: int, name: str) -> None:
        self.objects[name.lower()] = object_id

    def get_object(self, name: str) -> Optional[int]:
        return self.objects.get(name.lower())

    def get_path_between(self, start_name: str, end_name: str) -> List[List[float]]:
        start = self.get_location(start_name)
        end = self.get_location(end_name)

        if start is None or end is None:
            return []

        goal = False
        checked = [start]
        frontier = []

        current_Node = PlannerNode(start)
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
                    next_Node = PlannerNode(neigh)
                    next_Node.path = current_Node.path.copy()
                    next_Node.path.append(neigh.center)
                    next_Node.g = current_Node.g + np.linalg.norm(next_Node.location.center - current_Node.location.center)
                    next_Node.f = next_Node.g + np.linalg.norm(end.center - neigh.center)

                    frontier.append(next_Node)

        return []

    @classmethod
    def create_default_world(cls):
        world = cls()

        # Create locations using the new Location class
        door = Location("Door", [0,0], [0.0, 0.5, 0.5])
        living_room = Location("LivingRoom", [3,0], [3.0, 0.5, 0.5])
        fridge = Location("Fridge", [1,-2], [1.0, -2.5, 0.5])
        stove = Location("Stove", [3,-2], [3.0, -2.5, 0.5])
        tv = Location("TV", [1,2], [1.0, 2.5, 0.5])

        # Add locations to world
        world.add_location(door)
        world.add_location(living_room)
        world.add_location(fridge)
        world.add_location(stove)
        world.add_location(tv)

        # Set up location relationships using the world's add_next_to method
        world.add_next_to("door", ["livingroom", "fridge", "tv"])
        world.add_next_to("livingroom", ["tv", "door", "fridge", "stove"])
        world.add_next_to("fridge", ["door", "livingroom", "stove"])
        world.add_next_to("stove", ["fridge", "livingroom"])
        world.add_next_to("tv", ["door", "livingroom"])

        return world

    def create_default_physical_objects(self):
        """Create physical objects in the PyBullet simulation.
        This should be called after PyBullet has been initialized."""

        # Create test cube
        half_size = [0.1, 0.1, 0.1]
        cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_size)
        cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_size)
        cube_id = p.createMultiBody(baseMass=0.01,
                                   baseCollisionShapeIndex=cube_collision,
                                   baseVisualShapeIndex=cube_visual,
                                   basePosition=[1.0, 0.0, 0.1])

        # Add cube to world objects
        self.add_object(cube_id, "cube")

        # Define the location and rotation for the entire shelf
        shelf_location = np.array([1.0, 3.0, 0.0])
        shelf_scale = 0.25

        shelf_orientation = p.getQuaternionFromEuler([0, 0, np.pi])
        
        # Collision shapes for the simplified planes (links)
        # Define half-extents for plates and supports
        plate_dims = np.array([3.0, 2.0, 0.02])*shelf_scale
        support_dims_side = np.array([1/3, 2.0, 6.0])*shelf_scale
        support_dims_back = np.array([3.0, 1/3, 6.0])*shelf_scale


        plate_pos = [
            [0, 0, 1.5*shelf_scale],  # Bottom shelf
            [0, 0, 3.0*shelf_scale],  # Middle shelf
            [0, 0, 4.5*shelf_scale]   # Top shelf
        ]
        support_pos = [
            np.array([-1.5, 0, 3.0])*shelf_scale,   # Left support
            np.array([1.5, 0, 3.0])*shelf_scale,    # Right support
            np.array([0, -5/6, 3.0])*shelf_scale    # Back support
        ]

        # Create collision shapes for the plates and supports
        collision_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[d/2 for d in plate_dims])
        collision_support_side = p.createCollisionShape(p.GEOM_BOX, halfExtents=[d/2 for d in support_dims_side])
        collision_support_back = p.createCollisionShape(p.GEOM_BOX, halfExtents=[d/2 for d in support_dims_back])
        
        empty_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[1,1,1],              
            rgbaColor=[0, 0, 0, 0]     # fully transparent
        )


        # Create the visual shape from your Blender mesh
        visual_shelf_id = p.createVisualShape(p.GEOM_MESH, fileName="./models/Shelf.obj", meshScale=[shelf_scale, shelf_scale, shelf_scale])

        # Create the multibody object
        # This is where we apply the location and orientation to the base
        shelf_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shelf_id,
            basePosition=shelf_location,            # Correctly position the entire shelf
            baseOrientation=shelf_orientation,      # Apply the rotation here
            
            # Add links for each collision box
            linkMasses=[0] * 6,
            linkCollisionShapeIndices=[
                collision_box_id, collision_box_id, collision_box_id,
                collision_support_side, collision_support_side, collision_support_back
            ],
            linkVisualShapeIndices=[empty_visual] * 6,
            linkPositions=plate_pos + support_pos,
            linkOrientations=[[0,0,0,1]] * 6,
            linkInertialFramePositions=[[0,0,0]] * 6,
            linkInertialFrameOrientations=[[0,0,0,1]] * 6,
            linkParentIndices=[0] * 6,
            linkJointTypes=[p.JOINT_FIXED] * 6,
            linkJointAxis=[[0, 0, 0]] * 6
        )
        self.add_object(shelf_body, "shelf")

        return self.objects
