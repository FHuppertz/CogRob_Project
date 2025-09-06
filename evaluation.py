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
from world_state_checker import WorldStateChecker


def load_model(model_name):
    """
    Load a model using its correct model factory based on the model name.
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        Model instance or None if model cannot be loaded
    """
    # Map model names to their respective platforms and types
    model_mapping = {
        "gpt-4.1": {
            "platform": ModelPlatformType.OPENAI,
            "type": "gpt-4.1"
        },
        "claude-3.5-sonnet": {
            "platform": ModelPlatformType.ANTHROPIC,
            "type": ModelType.CLAUDE_3_5_SONNET
        },
        "qwen3-coder-480b-a35b-instruct": {
            "platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            "type": "local"
        }
    }
    
    if model_name not in model_mapping:
        print(f"Unknown model: {model_name}")
        return None
    
    model_info = model_mapping[model_name]
    
    try:
        if model_name == "gpt-4.1":
            if os.environ.get("OPENAI_API_KEY"):
                return ModelFactory.create(
                    model_platform=model_info["platform"],
                    model_type=model_info["type"],
                    model_config_dict={"temperature": 0.5},
                )
            else:
                print("OPENAI_API_KEY not found in environment variables")
                return None
                
        elif model_name == "claude-3.5-sonnet":
            if os.environ.get("ANTHROPIC_API_KEY"):
                return ModelFactory.create(
                    model_platform=model_info["platform"],
                    model_type=model_info["type"],
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                    model_config_dict={"temperature": 0.5},
                )
            else:
                print("ANTHROPIC_API_KEY not found in environment variables")
                return None
                
        elif model_name == "qwen3-coder-480b-a35b-instruct":
            if os.environ.get("LOCAL_API_KEY") and os.environ.get("LOCAL_API_HOST"):
                return ModelFactory.create(
                    model_platform=model_info["platform"],
                    model_type=model_info["type"],
                    api_key=os.environ.get("LOCAL_API_KEY"),
                    url=os.environ.get("LOCAL_API_HOST"),
                    model_config_dict={"stream": True}
                )
            else:
                print("LOCAL_API_KEY or LOCAL_API_HOST not found in environment variables")
                return None
                
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None


# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

print(f"Loaded config:")
pprint(config)

num_iterations = config.get('num_iterations', 1)
models_config = config.get('models', [])
prompts_config = config.get('prompts', [])
conditions_config = config.get('conditions', [])

# If no models specified, run once with no model
if not models_config:
    models_config = [None]

# If no prompts specified, run once with a default prompt
if not prompts_config:
    prompts_config = ["Put away all the items into the shelf."]

for model_name in models_config:
    for iteration_index in range(num_iterations):
        # Create a simulation environment
        world = World.create_default_world()
        sim = SimulationEnvironment(world, time_step=0.0, real_time=False, headless=True)
        sim.world.create_default_physical_objects()
        
        # Initialize world state checker
        state_checker = WorldStateChecker(world, conditions_config)

        # Load model if specified
        if model_name:
            print(f"Loading model: {model_name}")
            model = load_model(model_name)
        else:
            model = None

        robot = Robot(sim, model)
        sim.add_subscriber(robot)

        # Run the simulation
        sim.step(200)

        # Loop through prompts from config
        for task_index, task_prompt in enumerate(prompts_config):
            print(f"Executing prompt {task_index + 1}/{len(prompts_config)}: {task_prompt}")
            
            # Record initial state before robot invocation
            state_checker.record_initial_state(task_index)
            
            # Skip empty prompts
            if not task_prompt.strip():
                continue

            # Invoke the robot's agent with the task prompt
            robot.invoke(task_prompt)

            # Check final state after robot invocation
            results = state_checker.check_final_state(task_index)
            print(f"State check results for task {task_index}:")
            pprint(results)

            # Delay at end
            sim.step(100)

        # Disconnect from PyBullet
        sim.disconnect()
