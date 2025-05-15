#!/usr/bin/env python3
"""
vLLM implementation for Atari-GPT project.
This module provides a vLLM-based agent that can play Atari games.
"""

import os
import json
import logging
import numpy as np
import cv2
from PIL import Image
import io
import base64
import re
from typing import List, Dict, Any, Tuple, Optional, Union

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Set up logging
logger = logging.getLogger("atari-gpt.vllm_agent")

class VllmAgent:
    """
    Agent that uses vLLM models to play Atari games.
    Compatible with the original Atari-GPT Agent class.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", model="vllm", system_message=None, env=None,
                 max_model_len=2048, gpu_memory_utilization=0.85, tensor_parallel_size=1):
        """
        Initialize the VllmAgent.

        Args:
            model_name: The model to use
            model: The model key (should be 'vllm')
            system_message: The system prompt to use
            env: The Gymnasium environment
            max_model_len: Maximum sequence length (reduce to save memory)
            gpu_memory_utilization: GPU memory utilization (0.0 to 1.0)
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install it with 'pip install vllm'.")

        self.model_key = model
        logger.info(f'Model Key: {self.model_key}')

        self.model_name = model_name
        logger.info(f'Model Name: {self.model_name}')

        self.messages = []
        self.system_message = system_message
        self.env = env
        self.action_space = self.env.action_space.n
        self.response = None

        # Memory optimization parameters
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size

        # Log memory settings
        logger.info(f"Memory settings: max_model_len={max_model_len}, "
                   f"gpu_memory_utilization={gpu_memory_utilization}, "
                   f"tensor_parallel_size={tensor_parallel_size}")

        # Initialize vLLM model
        try:
            logger.info(f"Initializing vLLM with model: {self.model_name}")

            # Check if we're using a smaller model to adapt to memory constraints
            if "72B" in self.model_name and self.max_model_len > 4096:
                logger.warning(f"72B model detected with max_model_len={self.max_model_len}. "
                              f"This may require significant GPU memory. Consider reducing max_model_len.")

            # Initialize with memory optimization parameters
            self.llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 1},  # Limit to 1 image per prompt
                max_model_len=self.max_model_len,  # Reduce context length to save memory
                gpu_memory_utilization=self.gpu_memory_utilization,  # Use more GPU memory
                tensor_parallel_size=self.tensor_parallel_size  # Use tensor parallelism if multiple GPUs
            )
            logger.info("vLLM client initialized successfully.")

            # Prepare JSON schema for guided decoding
            self.json_schema = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "action": {"type": "integer", "minimum": 0, "maximum": self.action_space - 1}
                },
                "required": ["reasoning", "action"]
            }
            logger.info("JSON schema for vLLM guided decoding prepared.")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {str(e)}")
            raise

    def encode_image(self, cv_image):
        """
        Encode a CV2 image to base64.

        Args:
            cv_image: OpenCV image

        Returns:
            Base64 encoded string
        """
        _, buffer = cv2.imencode(".jpg", cv_image)
        return base64.b64encode(buffer).decode("utf-8")

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

    def add_user_message(self, frame=None, user_msg=None):
        """
        Add a user message to the conversation history.

        Args:
            frame: The image to include in the message (numpy array)
            user_msg: The text message from the user
        """
        if frame is not None and user_msg is not None:
            # Convert numpy array to PIL Image
            if isinstance(frame, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                pil_image = frame

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

    def add_assistant_message(self, demo_str=None):
        """
        Add an assistant message to the conversation history.

        Args:
            demo_str: Optional demo string to use instead of the response
        """
        if demo_str is not None:
            self.messages.append({
                "role": "assistant",
                "content": demo_str
            })
            return

        if self.response is not None:
            self.messages.append({
                "role": "assistant",
                "content": self.response
            })

    def delete_messages(self):
        """
        Delete old messages to maintain a context window.
        Compatible with the original Agent.delete_messages method.
        Uses a fixed context window size of 8 messages, matching API models.
        """
        logger.info('Deleting Set of Messages...')
        
        # Standard API model behavior: keep at most 8 messages
        message_len = 8

        if len(self.messages) >= message_len:
            if any(m["role"] == "system" for m in self.messages):
                # Keep system messages
                system_messages = [m for m in self.messages if m["role"] == "system"]
                # Keep most recent messages (excluding system messages)
                non_system_messages = [m for m in self.messages if m["role"] != "system"]
                # Calculate how many non-system messages to keep
                keep_count = message_len - len(system_messages)
                keep_count = max(0, min(keep_count, len(non_system_messages)))
                # Keep the most recent non-system messages
                recent_messages = non_system_messages[-keep_count:] if keep_count > 0 else []
                # Combine system and recent messages
                self.messages = system_messages + recent_messages
            else:
                # If no system messages, just keep the most recent messages
                self.messages = self.messages[-message_len:]
                
        logger.info(f"Context window maintained at {message_len} messages maximum")

    def query_LLM(self):
        """
        Query the LLM model.

        Returns:
            The model's response
        """
        # This is a placeholder - the actual response is generated in generate_response
        return "placeholder"

    def get_response(self):
        """
        Get a response from the model.

        Returns:
            The model's response
        """
        # This is a placeholder - the actual response is generated in generate_response
        return "placeholder"

    def clean_response(self, response, path):
        """
        Clean the model's response.

        Args:
            response: The model's response
            path: Path to save response data

        Returns:
            Cleaned response text
        """
        # For vLLM, we handle this in generate_response
        return response

    def check_action(self, response_text):
        """
        Extract and validate the action from the response.

        Args:
            response_text: The response text

        Returns:
            Valid action integer or 0 if invalid
        """
        try:
            # Try to parse as JSON
            response_json = json.loads(response_text)
            action = response_json.get("action")

            # Validate action
            if action is not None and 0 <= action < self.action_space:
                return action
        except:
            # If JSON parsing fails, try to extract action using regex
            match = re.search(r'"action":\s*(\d+)', response_text)
            if match:
                action = int(match.group(1))
                if 0 <= action < self.action_space:
                    return action

        # Default action if extraction fails
        return 0

    def close(self):
        """
        Close the vLLM agent and release resources.
        This is important to call between games to prevent resource leaks.
        """
        logger.info("Closing vLLM agent and releasing resources")
        # Delete references to free memory
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            self.llm = None

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
        except ImportError:
            pass

    def generate_response(self, path="./"):
        """
        Generate a response from the model.

        Args:
            path: Path to save response data

        Returns:
            Tuple of (action, full_response)
        """
        try:
            logger.info("Generating vLLM response...")

            # Prepare messages
            messages = []
            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})
            messages.extend(self.messages)

            # Sampling parameters with increased diversity
            sampling_params = SamplingParams(
                temperature=0.8,       # Higher temperature for more diverse responses
                top_p=0.95,            # Keep high top_p
                max_tokens=512,        # Keep high max_tokens
                frequency_penalty=0.7,  # Add frequency penalty to discourage repetition
                presence_penalty=0.7    # Add presence penalty to encourage new tokens
            )

            # Try different methods to generate a response
            try:
                # Method 1: Try using generate() method with formatted prompt
                prompt = f"System: {self.system_message}\n\n"
                for msg in self.messages:
                    role = msg["role"]
                    if isinstance(msg["content"], list):
                        # Multimodal message, extract text parts
                        text_parts = [item["text"] for item in msg["content"] if item["type"] == "text"]
                        content = " ".join(text_parts)
                        prompt += f"{role.capitalize()}: {content}\n"
                    else:
                        # Text-only message
                        prompt += f"{role.capitalize()}: {msg['content']}\n"

                # Add a JSON completion hint that encourages diversity
                prompt += "Assistant: I'll analyze this specific frame carefully and provide a unique response.\n"

                # Add a random seed to encourage diversity
                import random
                random_seed = random.randint(1, 10000)
                prompt += f"Random seed: {random_seed}\n"

                # Add game-specific hint based on environment name
                game_name = self.env.__class__.__name__
                if "Breakout" in game_name:
                    prompt += "I notice this is Breakout. I'll focus on the ball's trajectory and paddle position.\n"
                elif "Pong" in game_name:
                    prompt += "I notice this is Pong. I'll focus on the ball's direction and opponent's paddle.\n"
                elif "Alien" in game_name:
                    prompt += "I notice this is Alien. I'll focus on enemy positions and safe paths.\n"
                elif "Frogger" in game_name:
                    prompt += "I notice this is Frogger. I'll focus on avoiding obstacles and finding safe paths.\n"

                # Start the JSON without providing too much structure
                prompt += "My JSON response for this specific frame is:\n{"

                # Generate response
                outputs = self.llm.generate(
                    prompt,
                    sampling_params=sampling_params
                )

                # Get raw response
                response_text = outputs[0].outputs[0].text
                logger.info(f"Raw response: {response_text}")

                # Check if the response contains additional text after JSON
                # This happens when the model adds explanations after the JSON
                json_match = re.search(r'(\{.*?\})', response_text, re.DOTALL)
                if json_match:
                    json_part = json_match.group(1)
                    try:
                        # Try to parse the JSON part
                        json.loads(json_part)
                        # If successful, use only the JSON part
                        logger.info(f"Found valid JSON in response: {json_part}")
                        response_text = json_part
                    except json.JSONDecodeError:
                        # If parsing fails, continue with the full response
                        pass

                # Complete the JSON if it's incomplete
                if not response_text.endswith("}"):
                    # We started with {"reasoning": "
                    # So we need to complete it with ", "action": X}
                    if '"action"' not in response_text:
                        # Add action field if missing
                        response_text += "\", \"action\": 0}"
                    else:
                        # Just close the JSON if action is already there
                        response_text += "}"

                    logger.info(f"Completed JSON: {response_text}")

                # Remove markdown code block markers if present
                if "```json" in response_text:
                    response_text = response_text.replace("```json", "")
                    response_text = response_text.replace("```", "")
                    logger.info(f"Removed markdown code block markers: {response_text}")

                # Fix common formatting issues
                # Check if response starts with text without JSON structure or with "reasoning":
                if not response_text.strip().startswith("{") or response_text.strip().startswith('"reasoning":'):
                    # This is likely a direct text response like "Player should move right to avoid enemy",]
                    # Extract the text and action parts
                    reasoning_match = re.match(r'^(.*?)(?:,\s*\]|\",\s*|\"\s*,\s*)"action":\s*(\d+)', response_text, re.DOTALL)

                    if reasoning_match:
                        reasoning_text = reasoning_match.group(1).strip()
                        action_value = reasoning_match.group(2).strip()

                        # Clean up reasoning text - remove any "reasoning": prefix
                        reasoning_text = re.sub(r'^"?reasoning"?:\s*"?', '', reasoning_text)
                        # Remove trailing quotes if present
                        reasoning_text = re.sub(r'"$', '', reasoning_text)

                        # Create a proper JSON
                        response_text = f'{{"reasoning": "{reasoning_text}", "action": {action_value}}}'
                        logger.info(f"Reconstructed JSON from text: {response_text}")
                    else:
                        # If we can't extract reasoning and action, try to extract just the action
                        action_match = re.search(r'"action":\s*(\d+)', response_text)
                        if action_match:
                            action_value = action_match.group(1).strip()
                            # Get some text as reasoning
                            reasoning_text = response_text.split('"action":')[0].strip()

                            # Clean up reasoning text
                            reasoning_text = re.sub(r'^"?reasoning"?:\s*"?', '', reasoning_text)
                            if reasoning_text.endswith('",') or reasoning_text.endswith('",]'):
                                reasoning_text = reasoning_text[:-2]
                            if reasoning_text.startswith('"') and reasoning_text.endswith('"'):
                                reasoning_text = reasoning_text[1:-1]

                            # Create a proper JSON
                            response_text = f'{{"reasoning": "{reasoning_text}", "action": {action_value}}}'
                            logger.info(f"Extracted action and created JSON: {response_text}")
                else:
                    # Standard JSON fixes
                    # Fix issue with quotes followed by comma and bracket
                    response_text = re.sub(r'",\s*\]', '"}', response_text)
                    # Fix issue with missing quotes
                    response_text = re.sub(r'",\s*"action"', '", "action"', response_text)
                    # Fix issue with extra brackets
                    response_text = re.sub(r'\]\s*"action"', ', "action"', response_text)

                logger.info(f"After fixing formatting issues: {response_text}")

                # Remove any leading/trailing text that's not part of the JSON
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
                    logger.info(f"Extracted JSON: {response_text}")

                # Add randomness to prevent getting stuck in repetitive patterns
                # Every 10 steps, force a different action
                import random
                if hasattr(self, 'step_counter'):
                    self.step_counter += 1
                else:
                    self.step_counter = 0

                if self.step_counter % 10 == 0:
                    # Extract current action if possible
                    try:
                        current_json = json.loads(response_text)
                        current_action = current_json.get("action", 0)

                        # Choose a different action
                        available_actions = list(range(self.action_space))
                        if current_action in available_actions:
                            available_actions.remove(current_action)

                        if available_actions:
                            new_action = random.choice(available_actions)
                            # Replace the action in the JSON
                            current_json["action"] = new_action
                            current_json["reasoning"] = f"{current_json.get('reasoning', '')} [Exploring alternative action {new_action}]"
                            response_text = json.dumps(current_json)
                            logger.info(f"Forced exploration: changed action to {new_action}")
                    except:
                        pass

                # Try to fix common JSON errors
                try:
                    json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error: {str(e)}")
                    # Try to fix common errors
                    if "Expecting '\"' delimiter" in str(e):
                        # Fix missing quotes
                        response_text = re.sub(r'(\w+):', r'"\1":', response_text)
                        logger.info(f"Fixed missing quotes: {response_text}")

                    if "Expecting ':' delimiter" in str(e):
                        # Fix missing colons
                        response_text = re.sub(r'"(\w+)"\s+', r'"\1": ', response_text)
                        logger.info(f"Fixed missing colons: {response_text}")

                    # Fix trailing commas
                    response_text = re.sub(r',\s*}', '}', response_text)
                    logger.info(f"Fixed trailing commas: {response_text}")

                    # Fix missing braces
                    if not response_text.startswith("{"):
                        response_text = "{" + response_text
                        logger.info(f"Added opening brace: {response_text}")

                    if not response_text.endswith("}"):
                        response_text = response_text + "}"
                        logger.info(f"Added closing brace: {response_text}")

                    # Try to ensure action field exists
                    if '"action"' not in response_text:
                        # Add action field before the closing brace
                        response_text = response_text.rstrip("}") + ', "action": 0}'
                        logger.info(f"Added missing action field: {response_text}")

                    # Last resort: if we still can't parse it, create a minimal valid JSON
                    try:
                        json.loads(response_text)
                    except json.JSONDecodeError:
                        logger.warning("Still can't parse JSON, creating minimal valid JSON")
                        response_text = '{"reasoning": "Failed to parse response", "action": 0}'
                        logger.info(f"Created minimal valid JSON: {response_text}")

                # Store response for add_assistant_message
                self.response = response_text

                # Log the raw response for debugging
                logger.info(f"Raw response (first 100 chars): {response_text[:100]}...")

                # Try multiple parsing methods to extract action
                action = None

                # Method 1: Try to extract action from the beginning of the response
                # This handles cases where the model outputs "0" or "1" at the start
                start_action_match = re.match(r'^\s*(\d+)', response_text)
                if start_action_match:
                    try:
                        potential_action = int(start_action_match.group(1))
                        if 0 <= potential_action < self.action_space:
                            action = potential_action
                            logger.info(f"Successfully extracted action {action} from response start")
                    except ValueError:
                        pass

                # Method 2: Try to parse as JSON
                if action is None:
                    try:
                        # Try to parse the entire response as JSON
                        try:
                            response_json = json.loads(response_text)
                            if "action" in response_json and isinstance(response_json["action"], (int, str)):
                                potential_action = int(response_json["action"])
                                if 0 <= potential_action < self.action_space:
                                    action = potential_action
                                    logger.info(f"Successfully extracted action {action} from full JSON")
                        except json.JSONDecodeError:
                            # Try to extract JSON object using regex
                            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                            if json_match:
                                try:
                                    json_str = json_match.group(0)
                                    response_json = json.loads(json_str)
                                    if "action" in response_json and isinstance(response_json["action"], (int, str)):
                                        potential_action = int(response_json["action"])
                                        if 0 <= potential_action < self.action_space:
                                            action = potential_action
                                            logger.info(f"Successfully extracted action {action} from regex JSON")
                                except (json.JSONDecodeError, ValueError):
                                    pass
                    except Exception as e:
                        logger.warning(f"Error parsing JSON: {str(e)}")

                # Method 3: Try to extract action using regex pattern
                if action is None:
                    action_match = re.search(r'"action"\s*:\s*(\d+)', response_text)
                    if action_match:
                        try:
                            potential_action = int(action_match.group(1))
                            if 0 <= potential_action < self.action_space:
                                action = potential_action
                                logger.info(f"Successfully extracted action {action} from regex pattern")
                        except ValueError:
                            pass

                # Method 4: Look for action numbers in the text
                if action is None:
                    # Look for "action: X" or "Action: X" patterns
                    action_match = re.search(r'[aA]ction\s*:?\s*(\d+)', response_text)
                    if action_match:
                        try:
                            potential_action = int(action_match.group(1))
                            if 0 <= potential_action < self.action_space:
                                action = potential_action
                                logger.info(f"Successfully extracted action {action} from text pattern")
                        except ValueError:
                            pass

                # If all methods fail, use a default action
                if action is None:
                    # Use a more intelligent default: if we're playing Pong, use UP (2) or DOWN (3)
                    # instead of always using NOOP (0)
                    if "Pong" in self.env.__class__.__name__:
                        # Alternate between UP and DOWN
                        import random
                        action = random.choice([2, 3])
                        logger.warning(f"Could not extract valid action, using intelligent default {action} for Pong")
                    else:
                        action = 0
                        logger.warning(f"Could not extract valid action, using default action {action}")

                return action, response_text

            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return 0, str(e)

        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            return 0, str(e)
