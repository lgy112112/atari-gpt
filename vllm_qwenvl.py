#!/usr/bin/env python3
"""
vLLM implementation of Qwen2-VL model for Atari-GPT style gameplay.
This module provides a vLLM-based agent that can play Atari games using the Qwen2-VL model.
"""

import os
import json
import logging
import argparse
import numpy as np
import cv2
import gymnasium as gym
from gymnasium.wrappers import OrderEnforcing
from PIL import Image
import io
import base64
from tqdm import tqdm
import csv
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Tuple, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vllm_qwen.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("atari-gpt.vllm_qwen")

class ContinuousRecordVideo(gym.Wrapper):
    """
    Wrapper to record video of gameplay.
    Adapted from Atari-GPT project.
    """
    def __init__(self, env, video_folder, name_prefix=""):
        super().__init__(env)
        self.video_folder = video_folder
        self.name_prefix = name_prefix
        self.video_recorder = None
        os.makedirs(video_folder, exist_ok=True)

    def start_video_recorder(self):
        """Start the video recorder."""
        self.close_video_recorder()

        video_path = os.path.join(self.video_folder, f"{self.name_prefix}.mp4")
        self.video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(
            env=self.env,
            path=video_path,
            metadata={"title": self.name_prefix}
        )

    def close_video_recorder(self):
        """Close the video recorder."""
        if self.video_recorder:
            self.video_recorder.close()
            self.video_recorder = None

    def step(self, action):
        """Step the environment and record video frame."""
        observation, reward, terminated, truncated, info = self.env.step(action)

        if self.video_recorder:
            self.video_recorder.capture_frame()

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment."""
        observation, info = self.env.reset(**kwargs)

        if self.video_recorder:
            self.video_recorder.capture_frame()

        return observation, info

    def close(self):
        """Close the environment and video recorder."""
        self.close_video_recorder()
        return self.env.close()

class QwenVLAgent:
    """
    Agent that uses Qwen2-VL model to play Atari games.
    """
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", system_message=None, env=None):
        """
        Initialize the QwenVLAgent.

        Args:
            model_name: The model to use
            system_message: The system prompt to use
            env: The Gymnasium environment
        """
        self.model_name = model_name
        logger.info(f'Model Name: {self.model_name}')

        self.messages = []
        self.system_message = system_message
        self.env = env
        self.action_space = self.env.action_space.n

        # Initialize vLLM model
        try:
            logger.info(f"Initializing Qwen VLLM with model: {self.model_name}")
            self.llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 1}  # Limit to 1 image per prompt
            )
            logger.info("Qwen VLLM client initialized successfully.")

            # Prepare JSON schema for guided decoding
            self.json_schema = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "action": {"type": "integer", "minimum": 0, "maximum": self.action_space - 1}
                },
                "required": ["reasoning", "action"]
            }
            logger.info("JSON schema for Qwen VLLM guided decoding prepared.")

        except Exception as e:
            logger.error(f"Failed to initialize Qwen VLLM: {str(e)}")
            raise

    def add_user_message(self, image=None, user_msg=None):
        """
        Add a user message to the conversation history.

        Args:
            image: The image to include in the message (numpy array)
            user_msg: The text message from the user
        """
        if image is not None and user_msg is not None:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image

            # Add message with image
            self.messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_msg},
                    {"type": "image_url", "image_url": {"url": self._pil_to_data_url(pil_image)}}
                ]
            })
        elif user_msg is not None:
            # Text-only message
            self.messages.append({
                "role": "user",
                "content": user_msg
            })

    def add_assistant_message(self, content=None):
        """
        Add an assistant message to the conversation history.

        Args:
            content: The content of the assistant's message
        """
        if content is None:
            content = self.response

        self.messages.append({
            "role": "assistant",
            "content": content
        })

    def delete_messages(self, keep_last=10):  # 增加默认窗口长度
        """
        Delete old messages to maintain a context window and limit image count.

        Args:
            keep_last: Number of most recent messages to keep
        """
        # 首先保留系统消息和最近的keep_last条消息
        if len(self.messages) > keep_last:
            system_messages = [m for m in self.messages if m["role"] == "system"]
            recent_messages = self.messages[-keep_last:]
            self.messages = system_messages + recent_messages

        # 然后确保只保留最新的一个图像消息
        image_messages = []
        for i, msg in enumerate(self.messages):
            if msg["role"] == "user" and isinstance(msg["content"], list):
                # 检查是否包含图像
                has_image = any(item.get("type") == "image_url" for item in msg["content"])
                if has_image:
                    image_messages.append(i)

        # 如果有多个图像消息，只保留最新的一个
        if len(image_messages) > 1:
            logger.info(f"Found {len(image_messages)} image messages, keeping only the latest one")
            # 保留最后一个图像消息
            keep_image_idx = image_messages[-1]
            # 从其他图像消息中移除图像部分
            for idx in image_messages[:-1]:
                # 只保留文本部分
                text_parts = [item for item in self.messages[idx]["content"] if item.get("type") == "text"]
                if text_parts:
                    self.messages[idx]["content"] = text_parts
                else:
                    # 如果没有文本部分，将内容设为空字符串
                    self.messages[idx]["content"] = "Previous image message (image removed)"

    def _pil_to_data_url(self, pil_image):
        """
        Convert PIL image to data URL.

        Args:
            pil_image: PIL Image object

        Returns:
            Data URL string
        """
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def generate_response(self, save_path="./"):
        """
        Generate a response from the model.

        Args:
            save_path: Path to save response data

        Returns:
            Tuple of (action, full_response)
        """
        try:
            logger.info("[QWEN_VLLM_INFO] Generating response...")

            # 准备对话消息
            messages = []
            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})
            messages.extend(self.messages)

            # 使用基本的采样参数
            sampling_params = SamplingParams(
                temperature=1.0
            )

            try:
                # 尝试方法1: 使用messages参数
                logger.info("Attempting to use chat() with messages parameter")
                outputs = self.llm.chat(
                    messages=messages,
                    sampling_params=sampling_params
                )
            except Exception as e:
                logger.warning(f"chat() with messages parameter failed: {str(e)}")
                try:
                    # 尝试方法2: 不使用关键字参数
                    logger.info("Attempting to use chat() with positional parameter")
                    outputs = self.llm.chat(
                        messages,
                        sampling_params=sampling_params
                    )
                except Exception as e:
                    logger.warning(f"chat() with positional parameter failed: {str(e)}")

                    # 尝试方法3: 使用generate()方法
                    logger.info("Falling back to generate() method")
                    # 将对话转换为单一提示
                    prompt = f"System: {self.system_message}\n\n"
                    for msg in self.messages:
                        role = msg["role"]
                        if isinstance(msg["content"], list):
                            # 多模态消息，只提取文本部分
                            text_parts = [item["text"] for item in msg["content"] if item["type"] == "text"]
                            content = " ".join(text_parts)
                            prompt += f"{role.capitalize()}: {content}\n"
                        else:
                            # 纯文本消息
                            prompt += f"{role.capitalize()}: {msg['content']}\n"

                    prompt += "Assistant: "

                    # 使用generate方法
                    outputs = self.llm.generate(
                        prompt,
                        sampling_params=sampling_params
                    )

            response_text = outputs[0].outputs[0].text
            logger.info(f"Raw response: {response_text}")

            # 尝试解析响应
            try:
                import re

                # 方法1: 尝试从响应开头提取数字（处理类似"4 {"reasoning":...}"的格式）
                action_match = re.match(r'^\s*(\d+)', response_text)
                if action_match:
                    action = int(action_match.group(1))
                    if 0 <= action < self.action_space:
                        logger.info(f"Successfully extracted action {action} from response start")
                        return action, response_text

                # 方法2: 尝试直接解析整个响应为JSON
                try:
                    response_json = json.loads(response_text)
                    action = response_json.get("action")

                    # 验证动作
                    if action is not None and 0 <= action < self.action_space:
                        logger.info(f"Successfully extracted action {action} from JSON")
                        return action, response_text
                except json.JSONDecodeError:
                    pass

                # 方法3: 尝试使用正则表达式提取JSON部分
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        response_json = json.loads(json_str)
                        action = response_json.get("action")

                        # 验证动作
                        if action is not None and 0 <= action < self.action_space:
                            logger.info(f"Successfully extracted action {action} from JSON substring")
                            return action, response_text
                    except json.JSONDecodeError:
                        pass

                # 方法4: 尝试直接从文本中提取action字段
                action_match = re.search(r'"action":\s*(\d+)', response_text)
                if action_match:
                    action = int(action_match.group(1))
                    if 0 <= action < self.action_space:
                        logger.info(f"Successfully extracted action {action} from action field")
                        return action, response_text

                # 如果所有方法都失败，使用默认动作
                logger.warning(f"Could not extract valid action from response, using default action 0")
                return 0, response_text

            except Exception as e:
                logger.error(f"Error parsing response: {str(e)}")
                return 0, response_text

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return 0, str(e)

def run_test(env_name, prompt, output_dir="./experiments/"):
    """
    Run a test with the QwenVLAgent on an Atari game.

    Args:
        env_name: Name of the Atari environment
        prompt: System prompt for the agent
        output_dir: Directory to save experiment results
    """
    # Setup variables
    rewards = 0
    cum_rewards = []
    action_list = []
    header = ["actions", "cumulative_rewards"]

    # Get rid of ALE/ for creating folder
    if "ALE/" in env_name:
        temp_env_name = env_name[4:]
    else:
        temp_env_name = env_name

    # Create new experiment folders path with model name
    model_name = "qwen_vllm"
    new_dir = os.path.join(output_dir, temp_env_name[:-3] + '_' + model_name + '/')

    # Create folders if they do not exist
    os.makedirs(os.path.dirname(new_dir), exist_ok=True)

    # Check if the environment state is saved
    if os.path.exists(new_dir + 'env_' + temp_env_name[:-3] + '_state.pkl'):
        print('\n\nEnvironment Results Already Exist, Going to Next Environment...\n\n')
        return

    # Create Environment
    temp_env = gym.make(env_name, render_mode="rgb_array")

    # Apply the OrderEnforcer wrapper
    temp_env = OrderEnforcing(temp_env, disable_render_order_enforcing=True)

    # Reset the environment before any rendering
    temp_env.reset()

    # Record video
    env = ContinuousRecordVideo(env=temp_env, video_folder=new_dir, name_prefix=temp_env_name[:-3] + "_rollout")

    # Initialize agent
    agent = QwenVLAgent(model_name="Qwen/Qwen2-VL-2B-Instruct", system_message=prompt, env=env)

    # Start the recorder
    env.start_video_recorder()

    # Reset environment
    observation, info = env.reset()

    # Define parameters
    num_timesteps = 1000  # Same as original Atari-GPT project
    pause = 15
    buffer_pause = 19
    steps_taken = 0

    # User message template
    usr_msg = 'Analyze this game frame and select the optimal action. Focus on immediate gameplay elements visible in this specific frame, and follow the format: {"reasoning": "detailed step-by-step analysis", "action": X}'

    # Progress bar
    progress_bar = tqdm(total=num_timesteps, desc=f"Qwen-VL Rollout ({temp_env_name})", unit="steps")

    # Main loop
    for n in range(num_timesteps):
        observation = cv2.resize(observation, (512, 512))

        if n % 2 == 0:
            # Perform no-op action on even steps
            action = 0
            action_list.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            rewards += reward
            cum_rewards.append(rewards)

            if terminated or truncated:
                observation, info = env.reset()
        else:
            # Create buffer of 4 frames
            if n < buffer_pause:
                # Add frame and reason
                agent.add_user_message(observation, usr_msg)

                # Get response from model with action
                action, full_response = agent.generate_response(new_dir)

                # Add models reasoning to context
                agent.add_assistant_message(full_response)

                # Save action
                action_list.append(action)

                # Perform Action
                observation, reward, terminated, truncated, info = env.step(action)
                env.render()

                # Sum reward and save
                rewards += reward
                cum_rewards.append(rewards)

                # Check done condition
                if terminated or truncated:
                    observation, info = env.reset()
            else:
                # Add frame and reason
                agent.add_user_message(observation, usr_msg)

                # Have model reason from the given image
                action, full_response = agent.generate_response(new_dir)

                # Add models reasoning to context
                agent.add_assistant_message(full_response)

                # Save action
                action_list.append(action)

                # Perform Action
                observation, reward, terminated, truncated, info = env.step(action)
                env.render()

                # Sum reward and save
                rewards += reward
                cum_rewards.append(rewards)

                # Context buffer of only the 4 most recent frames
                # delete oldest context
                agent.delete_messages()

                # Check done condition
                if terminated or truncated:
                    observation, info = env.reset()

        steps_taken += 1
        progress_bar.update(1)
        progress_bar.set_postfix({"reward": rewards})

    # Close progress bar
    progress_bar.close()

    # Close the environment recorder
    env.close_video_recorder()

    # Close the environment
    env.close()

    # Save results
    with open(new_dir + 'actions_rewards.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for action, cum_reward in zip(action_list, cum_rewards):
            writer.writerow([action, cum_reward])

    print(f"\nTest completed. Final reward: {rewards}")
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Qwen-VL on Atari games')
    parser.add_argument('--game', type=str, default="PongDeterministic-v4",
                        help='Game to run (default: PongDeterministic-v4)')
    parser.add_argument('--output_dir', default='./experiments',
                        help='Directory to save experiment results')
    args = parser.parse_args()

    # Load environment configurations
    try:
        with open('envs.json') as file:
            environment_list = json.load(file)
    except FileNotFoundError:
        logger.error("envs.json file not found. Please make sure it exists in the current directory.")
        environment_list = {
            "PongDeterministic-v4": "You are a game playing assistant and will be provided an image. This will be of the game Pong, your goal is to provide me with what you believe to be the best action I could take to beat the game. Think about all possible actions and why each action is or is not the best action to take. The potential actions I can take are '0': NOOP '1': FIRE '2': RIGHT '3': LEFT '4': RIGHTFIRE '5': LEFTFIRE. Provide output as a json structured as {reasoning: reasoning for actions and why to choose an action, action: The environment action which would provide the best next state}. The action key should only have the action I should take for the current frame as a number."
        }

    # Get prompt for the specified game
    game = args.game
    if game in environment_list:
        prompt = environment_list[game]
        run_test(game, prompt, args.output_dir)
    else:
        logger.error(f"Game {game} not found in environment list")