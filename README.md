# CogRob_Project

## Project Overview

This project implements a cognitive robotics system where a Large Language Model (LLM) serves as the core decision-making component for a simulated mobile manipulator operating in a household environment. The agent is embodied in a PyBullet-based simulation as a mobile manipulator with an omnidirectional base and a 7-DoF robotic arm, capable of interacting with its environment through perception, reasoning, navigation, grasping, and placement actions.

The system demonstrates how LLMs can function as cognitive controllers for robots, enabling them to perform complex household tasks that require physical interaction through navigation and object manipulation. The agent's architecture includes:
- Working memory in the form of LLM context
- Episodic memory using ChromaDB for storing past experiences
- Tool-calling interface for high-level actions

## URDF Model Modifications

This project uses a modified version of the KUKA iiwa robot URDF model. Two key changes were made to the original model to make it suitable for mobile manipulation tasks:

1. **Base Link Mass Increase**: The mass of link_0 (base) was increased from 0.0 kg to 20.0 kg to provide a stable base for the robot arm when mounted on a mobile platform.

2. **First Joint Type Change**: Joint_1 was changed from a "revolute" joint with position limits to a "continuous" joint without position limits, allowing unlimited rotation which is essential for a mobile manipulator.

### Specific Changes

**File**: `model_2.urdf`

**Change 1 - Base Link Mass**:
- **Line 67**: `<mass value="0.0"/>` → `<mass value="20.0"/>`
- This increases the stability of the robot's base.

**Change 2 - First Joint Type**:
- **Line 85**: `<joint name="lbr_iiwa_joint_1" type="revolute">` → `<joint name="lbr_iiwa_joint_1" type="continuous">`
- **Lines 90**: Removed position limits:
  ```xml
  <limit effort="300" lower="-2.96705972839" upper="2.96705972839" velocity="10"/>
  ```
  Changed to:
  ```xml
  <limit effort="300" velocity="10"/>
  ```

These modifications are essential for the proper functioning of the mobile manipulator in simulation. The modified URDF can then be saved as `./models/model_2.urdf`.

### File Path

The modified URDF file needs to be saved in your environment's site-packages directory:
`./env/lib64/python3.12/site-packages/pybullet_data/kuka_iiwa/model_2.urdf`

## Prerequisites

To run this project, you need to install the following dependencies:

```bash
pip install pybullet numpy chromadb python-dotenv
```

Additionally, you need to install the CAMEL library for LLM integration:
```bash
pip install camel-ai
```

For the LLM to function properly, you'll need API keys for one of the supported platforms:
- OpenAI API key
- Anthropic API key
- Local LLM API (e.g., vLLM)

Set these in your environment variables or create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key
# OR
ANTHROPIC_API_KEY=your_anthropic_api_key
# OR for local models
LOCAL_API_KEY=your_local_api_key
LOCAL_API_HOST=http://localhost:8000/v1
```

## How to Run the Code

To run the simulation, execute the main script:

```bash
python simulation.py
```

When you run the simulation:
1. A PyBullet GUI window will open showing the environment
2. The robot will be initialized in the environment
3. An interactive command-line interface will appear where you can enter tasks for the robot

Example tasks you can give to the robot:
- "Move to the kitchen shelf"
- "Grab the mug from the table"
- "Place the box on the bottom shelf"

The robot will use its LLM-based cognitive architecture to interpret these commands and execute appropriate actions through its tool interface.

## Robot Capabilities and Tools

The robot has access to several tools that allow it to interact with its environment:

1. **Look around**: Returns information about the current environment, including locations and objects
2. **Move to**: Navigate to a specific location in the environment using path planning
3. **Grab**: Pick up an object by name if within reach
4. **Place**: Place a held object at a specified location
5. **Search memory**: Query past experiences stored in episodic memory
6. **Add to scratchpad**: Write down thoughts and plans for reasoning
7. **View scratchpad**: Review previous thoughts and plans
8. **End task**: Complete the current task with a status report

## Adding Objects to the World

To add new objects to the world, you can modify the `create_default_physical_objects()` method in `world.py`. Here's how to add a new object:

1. Create a visual shape for the object:
   ```python
   visual_shape = p.createVisualShape(p.GEOM_MESH, fileName="./models/YourObject.obj", meshScale=[scale, scale, scale])
   ```

2. Create a collision shape:
   ```python
   collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[x_size/2, y_size/2, z_size/2])
   ```

3. Create the multi-body object:
   ```python
   object_id = p.createMultiBody(
       baseMass=mass,
       baseCollisionShapeIndex=collision_shape,
       baseVisualShapeIndex=visual_shape,
       basePosition=[x, y, z],
       baseOrientation=p.getQuaternionFromEuler([roll, pitch, yaw])
   )
   ```

4. Add the object to the world's object dictionary:
   ```python
   self.add_object(object_id, "object_name")
   ```

## Adding New Locations

To add new locations to the semantic map:

1. Create a new Location object:
   ```python
   new_location = Location("Location Name", [x, y])
   ```

2. Add it to the world:
   ```python
   world.add_location(new_location)
   ```

3. Connect it to neighboring locations:
   ```python
   world.add_next_to("New Location", ["Existing Location 1", "Existing Location 2"])
   ```

## Memory System

The robot uses ChromaDB for episodic memory storage. Memories are stored as vector embeddings and can be retrieved using semantic search. The memory system allows the robot to learn from past experiences and adapt its strategies for similar future tasks.
