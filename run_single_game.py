#!/usr/bin/env python3
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
            results = run(args.game, prompt, args.model, output_dir=args.output_dir, vllm_model=args.vllm_model)
        else:
            results = run(args.game, prompt, args.model, output_dir=args.output_dir)

        logger.info(f"Completed single game: {args.game}")
    else:
        logger.error(f"Game {args.game} not found in environment list")

if __name__ == "__main__":
    main()
