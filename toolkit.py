from typing import List, Optional, TYPE_CHECKING, Union
from camel.toolkits import FunctionTool, BaseToolkit

if TYPE_CHECKING:
    from robot import Robot
    from memory import Memory

class RobotToolkit(BaseToolkit):
    """Toolkit for robot actions like moving, grabbing, placing objects, and memory operations."""

    def __init__(self, robot: 'Robot', memory: 'Memory'):
        """Initialize the toolkit with robot instance and memory.

        Args:
            robot: The Robot instance that will execute the actions
            memory: The Memory instance for storing and retrieving memories
        """
        self.robot = robot
        self.memory = memory
        super().__init__()

    def move_to(self, target: Union[str, List[float]]) -> dict:
        """Move to a target location.

        Args:
            target: Either a string name of a location or [x, y] coordinates.

        Returns:
            dict: Result of the movement action including status.
        """
        result = self.robot.move_to(target)
        return result

    def grab(self, target: Union[str, int]) -> dict:
        """Grab an object by name or ID. You must move to the location containing
        the object first before grabbing it. Grabbing an object makes the object
        the currently held object.

        Args:
            target: Either a string name of an object or a direct object ID.

        Returns:
            dict: Result of the grab action including status.
        """
        result = self.robot.grab(target)
        return result

    def place(self, location: str, place_position: Optional[str]) -> dict:
        """Place the currently held object at a target location. You must move to
        the location first before placing the object there, and must also have an
        object that is currently being held.

        Args:
            location: A string name of a location
            place_position: Optional position within the location to place the object

        Returns:
            dict: Result of the place action including status.
        """
        result = self.robot.place(location, place_position)
        return result

    def finish_task(self, status: str, description: str, summary: str) -> dict:
        """Finish the current task with a status report.

        Args:
            status: Whether the task execution was a success, a failure, or unknown.
            description: A description of the task, that is independent of what was actually done.
                This is a description of the task given to the robot.
            summary: A description of the execution trace of the robot to perform the task. Be
                sure to include any relevant details about the task execution in detail.

        Returns:
            dict: Result confirming the task completion.
        """
        result = {
            "status": status,
            "description": description,
            "summary": summary
        }
        print(f"Agent has finished task with result:\n{result}")
        
        # Store the task result in memory
        metadata = {"status": status, "summary": summary}
        self.memory.add_memory(description, metadata)
        
        return result

    def search_memory(self, query: str) -> dict:
        """Search the robot's memory for relevant past tasks.

        Args:
            query: A query string to search for in the memory

        Returns:
            dict: Search results with status and list of matching memories
        """
        # Search for memories using the query
        memories = self.memory.search_memories(query)
        
        # Format the results to be similar to finish_task output
        formatted_results = []
        for memory in memories:
            formatted_result = {
                "status": memory["metadata"].get("status", "unknown"),
                "description": memory["content"],
                "summary": memory["metadata"].get("summary", "")
            }
            formatted_results.append(formatted_result)
        
        return {
            "status": "success",
            "query": query,
            "results": formatted_results
        }

    def get_tools(self) -> list[FunctionTool]:
        """Get list of available tools."""
        return [
            FunctionTool(self.move_to),
            FunctionTool(self.grab),
            FunctionTool(self.place),
            FunctionTool(self.finish_task),
            FunctionTool(self.search_memory)
        ]
