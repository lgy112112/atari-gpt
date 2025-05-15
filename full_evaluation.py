import json
import argparse
import logging
import os
import sys
import subprocess
from run_experiments import run

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)

# Suppress httpx and urllib3 logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("atari-gpt")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Atari-GPT evaluations')
    parser.add_argument('--models', nargs='+', default=['rand', 'gpt4', 'gpt4o', 'gemini', 'claude', 'vllm'],
                        help='Models to evaluate (default: all models)')
    parser.add_argument('--games', nargs='+', default=None,
                        help='Specific games to evaluate (default: all games)')
    parser.add_argument('--output_dir', default='./experiments',
                        help='Directory to save experiment results')
    parser.add_argument('--vllm_model', default='Qwen/Qwen2.5-VL-3B-Instruct',
                        help='Model name for vLLM (default: Qwen/Qwen2.5-VL-3B-Instruct)')
    parser.add_argument('--vllm_model_list', action='store_true',
                        help='Show a list of recommended vLLM models and exit')
    parser.add_argument('--vllm_separate_processes', action='store_true',
                        help='Run each vLLM game in a separate process to avoid resource conflicts')
    parser.add_argument('--vllm_max_model_len', type=int, default=32768,
                        help='Maximum sequence length for vLLM (default: 32768, reduce to save memory)')
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.85,
                        help='GPU memory utilization for vLLM (default: 0.85, range: 0.0-1.0)')
    parser.add_argument('--vllm_tensor_parallel_size', type=int, default=1,
                        help='Number of GPUs for tensor parallelism (default: 1)')
    args = parser.parse_args()

    # Show vLLM model list if requested
    if args.vllm_model_list:
        print("\n=== Recommended vLLM Models for Atari Games ===")
        print("Models are listed from strongest to weakest:")
        print("1. Qwen/Qwen2-VL-72B-Instruct - Best performance, requires high-end GPU")
        print("2. Qwen/Qwen2-VL-7B-Instruct - Good balance of performance and resource usage")
        print("3. Qwen/Qwen2.5-VL-3B-Instruct - Fastest but less capable")
        print("4. llava-hf/llava-1.5-13b-hf - Strong alternative to Qwen")
        print("5. llava-hf/llava-1.5-7b-hf - Smaller LLaVA model")
        print("\nUsage example: python full_evaluation.py --models vllm --vllm_model Qwen/Qwen2-VL-7B-Instruct")
        sys.exit(0)

    # If 'all' is specified, replace it with the list of all models
    if 'all' in args.models:
        args.models = ['rand', 'gpt4', 'gpt4o', 'gemini', 'claude', 'vllm']

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load environment configurations
    try:
        with open('envs.json') as file:
            environment_list = json.load(file)
    except FileNotFoundError:
        logger.error("envs.json file not found. Please make sure it exists in the current directory.")
    except json.JSONDecodeError:
        logger.error("Error parsing envs.json. Please check the file format.")

    # Filter games if specified
    if args.games:
        environments = {k: v for k, v in environment_list.items() if k in args.games}
        if not environments:
            logger.error(f"None of the specified games {args.games} found in envs.json")
    else:
        environments = environment_list

    # Run evaluations for each model and game
    for model in args.models:
        logger.info(f"Starting evaluation for model: {model}")
        for game, prompt in environments.items():
            try:
                logger.info(f"Running test for: {game}")
                # Handle vLLM models differently if separate processes are requested
                if model == 'vllm' and args.vllm_separate_processes:
                    logger.info(f"Running vLLM game {game} in a separate process")
                    # Create a command to run a single game in a separate process
                    cmd = [
                        sys.executable,
                        "run_single_game.py",
                        "--game", game,
                        "--model", model,
                        "--output_dir", args.output_dir,
                        "--vllm_model", args.vllm_model,
                        "--vllm_max_model_len", str(args.vllm_max_model_len),
                        "--vllm_gpu_memory_utilization", str(args.vllm_gpu_memory_utilization),
                        "--vllm_tensor_parallel_size", str(args.vllm_tensor_parallel_size)
                    ]

                    # Create run_single_game.py if it doesn't exist
                    if not os.path.exists("run_single_game.py"):
                        with open("run_single_game.py", "w") as f:
                            f.write("""#!/usr/bin/env python3
import json
import argparse
import logging
import os
from run_experiments import run

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("single_game.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("atari-gpt-single")

def main():
    parser = argparse.ArgumentParser(description='Run a single Atari-GPT evaluation')
    parser.add_argument('--game', type=str, required=True, help='Game to evaluate')
    parser.add_argument('--model', type=str, required=True, help='Model to use')
    parser.add_argument('--output_dir', default='./experiments', help='Output directory')
    parser.add_argument('--vllm_model', default='Qwen/Qwen2.5-VL-3B-Instruct', help='vLLM model name')
    parser.add_argument('--vllm_max_model_len', type=int, default=32768, help='Maximum sequence length for vLLM')
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.85, help='GPU memory utilization for vLLM')
    parser.add_argument('--vllm_tensor_parallel_size', type=int, default=1, help='Number of GPUs for tensor parallelism')
    args = parser.parse_args()

    # Load environment configurations
    try:
        with open('envs.json') as file:
            environment_list = json.load(file)
    except FileNotFoundError:
        logger.error("envs.json file not found")
        return

    # Get prompt for the game
    if args.game in environment_list:
        prompt = environment_list[args.game]
        logger.info(f"Running single game: {args.game} with model: {args.model}")

        # Run the game
        if args.model == 'vllm':
            results = run(
                args.game,
                prompt,
                args.model,
                output_dir=args.output_dir,
                vllm_model=args.vllm_model,
                vllm_max_model_len=args.vllm_max_model_len,
                vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                vllm_tensor_parallel_size=args.vllm_tensor_parallel_size
            )
        else:
            results = run(args.game, prompt, args.model, output_dir=args.output_dir)

        logger.info(f"Completed single game: {args.game}")
    else:
        logger.error(f"Game {args.game} not found in environment list")

if __name__ == "__main__":
    main()
""")

                    # Run the command
                    process = subprocess.Popen(cmd)
                    process.wait()  # Wait for the process to complete
                    logger.info(f"Separate process for game {game} completed")

                # Run in the current process
                else:
                    # Pass vllm parameters if the model is vllm
                    if model == 'vllm':
                        results = run(
                            game,
                            prompt,
                            model,
                            output_dir=args.output_dir,
                            vllm_model=args.vllm_model,
                            vllm_max_model_len=args.vllm_max_model_len,
                            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size
                        )
                    else:
                        results = run(game, prompt, model, output_dir=args.output_dir)
                logger.info(f"Completed test for: {game}")
            except Exception as e:
                logger.error(f"Error running test for {game} with model {model}: {str(e)}")
                continue
        logger.info(f"Completed evaluation for model: {model}")

if __name__ == "__main__":
    main()

