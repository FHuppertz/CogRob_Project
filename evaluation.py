import dotenv
dotenv.load_dotenv()

import os
import yaml
import pybullet as p

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from pprint import pprint

from world import World
from robot import Robot
from simulation import SimulationEnvironment

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

print(f"Loaded config:")
pprint(config)

num_iterations = config.get('num_iterations', 1)

for iteration_index in range(num_iterations):
    # Create a simulation environment
    world = World.create_default_world()
    sim = SimulationEnvironment(world, time_step=0.0, real_time=False, headless=True)
    sim.world.create_default_physical_objects()

    # Add a subscriber (e.g., a robot)
    if os.environ.get("OPENAI_API_KEY", None) or os.getenv("ANTHROPIC_API_KEY") or os.getenv("VLLM_API_KEY"):
        # model = ModelFactory.create(
        #   model_platform=ModelPlatformType.OPENAI,
        #   model_type="gpt-4.1",
        #   model_config_dict={"temperature": 0.5},
        # )
        # model = ModelFactory.create(
        #     model_platform=ModelPlatformType.ANTHROPIC,
        #     model_type=ModelType.CLAUDE_3_5_SONNET,
        #     api_key=os.getenv("ANTHROPIC_API_KEY"),
        #     model_config_dict={
        #         # "temperature": 0.5,
        #         # "stream": True
        #     }
        # )
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="local",
            api_key=os.getenv("LOCAL_API_KEY"),
            url=os.getenv("LOCAL_API_HOST"),
            model_config_dict={
                # "temperature": 0.5,
                "stream": True
            }
        )
    else:
        model = None

    robot = Robot(sim, model)
    sim.add_subscriber(robot)

    # Run the simulation
    sim.step(200)

    # Interactive CLI for sending commands to the robot
    try:
        while True:
            # Get user input
            task_prompt = input("Enter a task for the robot (or 'quit' to exit): ")

            # Check if user wants to quit
            if task_prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting simulation...")
                break

            # Skip empty prompts
            if not task_prompt.strip():
                continue

            # Invoke the robot's agent with the user's task
            robot.invoke(task_prompt)

            # Delay at end
            sim.step(100)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

    # Disconnect from PyBullet
    sim.disconnect()
