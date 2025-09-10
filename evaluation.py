import dotenv
dotenv.load_dotenv()

import os
import yaml
import pybullet as p
import csv

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from pprint import pprint

from world import World
from robot import Robot
from simulation import SimulationEnvironment
from world_state_checker import WorldStateChecker
from logging_utils import get_logger

# Initialize logger
logger = get_logger("EVALUATION")


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
            "type": ModelType.GPT_4_1
        },
        "claude-3.5-sonnet": {
            "platform": ModelPlatformType.ANTHROPIC,
            "type": ModelType.CLAUDE_3_5_SONNET
        },
        "claude-sonnet-4": {
            "platform": ModelPlatformType.ANTHROPIC,
            "type": ModelType.CLAUDE_SONNET_4
        },
        "qwen3-coder-480b-a35b-instruct": {
            "platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            "type": "qwen3-coder-480b-a35b-instruct"
        },
        "gemini-2.5-pro": {
            "platform": ModelPlatformType.GEMINI,
            "type": ModelType.GEMINI_2_5_PRO
        },
        "gemini-2.5-flash": {
            "platform": ModelPlatformType.GEMINI,
            "type": ModelType.GEMINI_2_5_FLASH
        }
    }
    
    if model_name not in model_mapping:
        logger.warning(f"Unknown model: {model_name}")
        return None
    
    model_info = model_mapping[model_name]
    
    try:
        if model_name == "gpt-4.1":
            if os.environ.get("OPENAI_API_KEY"):
                return ModelFactory.create(
                    model_platform=model_info["platform"],
                    model_type=model_info["type"],
                    # model_config_dict={"temperature": 0.5},
                )
            else:
                logger.warning("OPENAI_API_KEY not found in environment variables")
                return None
                
        elif "claude" in model_name:
            if os.environ.get("ANTHROPIC_API_KEY"):
                return ModelFactory.create(
                    model_platform=model_info["platform"],
                    model_type=model_info["type"],
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                    # model_config_dict={"temperature": 0.5},
                )
            else:
                logger.warning("ANTHROPIC_API_KEY not found in environment variables")
                return None
                
        elif model_name == "qwen3-coder-480b-a35b-instruct":
            if os.environ.get("LOCAL_API_KEY") and os.environ.get("LOCAL_API_HOST"):
                return ModelFactory.create(
                    model_platform=model_info["platform"],
                    model_type=model_info["type"],
                    api_key=os.environ.get("LOCAL_API_KEY"),
                    url=os.environ.get("LOCAL_API_HOST"),
                    model_config_dict={
                        "stream": True,
                        "temperature": 0.7,
                        }
                )
            else:
                logger.warning("LOCAL_API_KEY or LOCAL_API_HOST not found in environment variables")
                return None
        
        elif "gemini" in model_name:
            if os.environ.get("GEMINI_API_KEY"):
                return ModelFactory.create(
                    model_platform=model_info["platform"],
                    model_type=model_info["type"],
                    model_config_dict={
                        "stream": True,
                        # "temperature": 0.7,
                        }
                )
            else:
                logger.warning("GEMINI_API_KEY not found in environment variables")
                return None
                
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return None

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

logger.info(f"Loaded config:")
pprint(config)

time_step = config.get('time_step', 0.01)
real_time = config.get('real_time', True)
headless = config.get('headless', False)
num_trials = config.get('num_trials', 1)
num_memory_trials = config.get('num_memory_trials', 2)
models_config = config.get('models', [])
prompts_config = config.get('prompts', [])
conditions_config = config.get('conditions', [])

# If no models specified, run once with no model
if not models_config:
    models_config = [None]

# If no prompts specified, run once with a default prompt
if not prompts_config:
    prompts_config = ["Put away all the items into the shelf."]

# Initialize CSV file handling
file_exists = os.path.exists('results.csv')
fieldnames = ['Model', 'Memory', 'Task', 'Trial', 'Invokes', 'Toolcalls', 'Belief', 'Truth', 'Accuracy', 'Stopped']

for model_name in models_config:
    for iteration_index in range(num_trials):
        for memory_trial_index in range(num_memory_trials):
            # Create a simulation environment
            world = World.create_default_world()
            sim = SimulationEnvironment(world, time_step=time_step, real_time=real_time, headless=headless)
            sim.world.create_default_physical_objects()

            # Initialize world state checker
            state_checker = WorldStateChecker(world, conditions_config)

            # Load model if specified
            if model_name:
                logger.info(f"Loading model: {model_name}")
                model = load_model(model_name)

                if model is None:
                    continue
            else:
                model = None
                continue

            robot = Robot(sim, model)
            if robot.chat_agent is not None:
                robot.chat_agent.init_messages()

                # Delete all memories if this is the first memory iteration
                if memory_trial_index == 0:
                    robot.memory.delete_all_memories()
 
            sim.add_subscriber(robot)

            # Run the simulation
            sim.step(200)

            # Loop through prompts from config
            for task_index, task_prompt in enumerate(prompts_config):
                logger.info(f"Executing prompt {task_index + 1}/{len(prompts_config)}: {task_prompt}")

                # Record initial state before robot invocation
                state_checker.record_initial_state(task_index)

                # Skip empty prompts
                if not task_prompt.strip():
                    continue

                # Reset toolkit counters before each task
                if robot.toolkit:
                    robot.toolkit.num_toolcalls = 0
                    robot.toolkit.end_task_status = None
                    robot.num_invokes = 0

                # Invoke the robot's agent with the task prompt
                robot.invoke(task_prompt)

                results = state_checker.check_final_state(task_index)

                # Extract data for CSV using dictionary
                result_dict = {
                    'Model': robot.chat_agent.model_backend.model_type if robot.chat_agent else "None",
                    'Memory': memory_trial_index,
                    'Task': task_index + 1,  # 1-based indexing for tasks
                    'Trial': iteration_index + 1,  # 1-based indexing for trials
                    'Invokes': robot.num_invokes,
                    'Toolcalls': robot.toolkit.num_toolcalls if robot.toolkit else 0,
                    'Belief': 1 if (robot.toolkit.end_task_status == "success" if robot.toolkit else False) else 0,
                    'Truth': 1 if (results.get('status', 'error') == "success") else 0,
                    'Accuracy': 0,  # Will be calculated after getting belief and truth
                    'Stopped': 1 if robot.stopped else 0,
                }

                # Calculate accuracy (1 if belief matches truth, 0 otherwise)
                result_dict['Accuracy'] = 1 if result_dict['Belief'] == result_dict['Truth'] else 0

                # Save result to CSV immediately after each iteration
                with open('results.csv', 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    # Only write header if file doesn't exist or is empty
                    if not file_exists or os.path.getsize('results.csv') == 0:
                        writer.writeheader()
                        file_exists = True  # Update flag after writing header
                    
                    writer.writerow(result_dict)

                logger.info(f"State check results for task {task_index}:")
                pprint(results)
                logger.info(f"Collected data: Model={result_dict['Model']}, Memory={result_dict['Memory']}, "
                      f"Task={result_dict['Task']}, Trial={result_dict['Trial']}, "
                      f"Toolcalls={result_dict['Toolcalls']}, Belief={result_dict['Belief']}, "
                      f"Truth={result_dict['Truth']}, Accuracy={result_dict['Accuracy']}, "
                      f"Stopped={result_dict['Stopped']}"
                      )

                # Delay at end
                sim.step(100)

                if not results.get('status', 'error') == "success":
                    logger.info(f"Stopping at task index {task_index} as robot was unsuccessful...")
                    break

            # Disconnect from PyBullet
            sim.disconnect()
