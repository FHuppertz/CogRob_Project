from typing import TYPE_CHECKING, Union, List
from camel.toolkits import FunctionTool, BaseToolkit
import numpy as np

if TYPE_CHECKING:
    from robot import Robot

class RobotToolkit(BaseToolkit):
    """Toolkit for robot actions like moving, grabbing, and placing objects."""

    def __init__(self, robot: 'Robot'):
        """Initialize the toolkit with robot instance.

        Args:
            robot: The Robot instance that will execute the actions
        """
        self.robot = robot
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

    def place(self, target: Union[str, List[float]]) -> dict:
        """Place the currently held object at a target location. You must move to 
        the location first before placing the object there, and must also have an
        object that is currently being held.

        Args:
            target: Either a string name of a location or [x, y, z] coordinates.

        Returns:
            dict: Result of the place action including status.
        """
        result = self.robot.place(target)
        return result

    def get_tools(self) -> list[FunctionTool]:
        """Get list of available tools."""
        return [
            FunctionTool(self.move_to),
            FunctionTool(self.grab),
            FunctionTool(self.place)
        ]
