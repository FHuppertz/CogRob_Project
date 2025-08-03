import numpy as np

from typing import Dict, List


class Location():
    def __init__(self, name, center, place_position=None):
        self.name = name
        self.center = np.array(center)
        # Default place position is slightly offset from center
        self.place_position = np.array(place_position) if place_position is not None else np.array([center[0], center[1], 0.5])
        self.neighbours = []

    def next_to(self, neighbours):
        for neighbour in neighbours:
            if neighbour not in self.neighbours:
                self.neighbours.append(neighbour)
                neighbour.next_to([self])


class PlannerNode():
    def __init__(self, location: Location):
        self.location = location
        self.f = 0
        self.g = 0
        self.path = []


class World():
    def __init__(self):
        self.locations: Dict[str, Location] = {}
        self.objects: Dict[str, int] = {}

    def add_location(self, location: Location):
        self.locations[location.name.lower()] = location

    def add_next_to(self, location: str, neighbours: List[str]):
        self.locations[location.lower()].next_to([self.locations[neighbour.lower()] for neighbour in neighbours])
        for neighbour in neighbours:
            neighbour_location = self.locations[neighbour.lower()]
            neighbour_location.next_to([self.locations[location.lower()]])

    def get_location(self, name):
        return self.locations.get(name.lower())

    def add_object(self, object_id: int, name: str):
            self.objects[name.lower()] = object_id

    def get_object(self, name):
        return self.objects.get(name.lower())

    def get_path_between(self, start_name, end_name):
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
