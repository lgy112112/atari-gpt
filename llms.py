from openai import OpenAI 
import anthropic 
import google.generativeai as genai 

import base64 
import cv2 
import json 
import re 
from google.generativeai.types import HarmCategory, HarmBlockThreshold 

class Agent(): 
    def __init__(self, model_name=None, model = None, system_message=None, env=None): 

        # Get model key for what api to call
        self.model_key = model 
        print('Model Key: ', self.model_key) 

        # Get the model name (for calling correct model to query)
        self.model_name = model_name 
        
        print('Model Name: ', self.model_name) 

        # Create list of messages
        self.messages = [] 

        # Get system prompt
        self.system_message = system_message 
        
        # Get env 
        self.env = env 

        # Get action space 
        self.action_space = self.env.action_space.n 

        self.reset_count = 0 

        # Set up correct model to call 
        if self.model_key == 'gpt4o' or self.model_key == 'gpt4': 
            file = open("OPENAI_API_KEY.txt", "r") 
            api_key = file.read() 
            self.client = OpenAI(api_key=api_key) 

            if system_message is not None: 
                system_prompt = {"role": "system", "content": [system_message]} 
                self.messages.append(system_prompt) 

        elif self.model_key == 'claude':
            file = open("ANTHROPIC_API_KEY.txt", "r")
            api_key = file.read()
            self.client = anthropic.Anthropic(api_key=api_key)
        
        elif self.model_key == 'gemini':
            file = open("GOOGLE_API_KEY.txt", "r")
            api_key = file.read()
            genai.configure(api_key=api_key)
            generation_config = genai.GenerationConfig(temperature=1)
            if self.system_message is not None:
                self.client = genai.GenerativeModel(model_name = self.model_name, system_instruction=self.system_message, generation_config=generation_config)
            else:
                self.client = genai.GenerativeModel(model_name = self.model_name, generation_config=generation_config)

    def encode_image(self, cv_image):
        _, buffer = cv2.imencode(".jpg", cv_image)
        return base64.b64encode(buffer).decode("utf-8")
    
    def query_LLM(self):

        # Check which model to use and prompt the model 
        if self.model_key=='gpt4' or self.model_key=='gpt4o':
            self.response = self.client.chat.completions.create(
                model=self.model_name,
                response_format={ "type": "json_object" },
                messages=self.messages,
                temperature=1,
            )

        elif self.model_key == 'claude':
            if self.system_message is not None:
                self.response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=1,
                    system=self.system_message,
                    messages=self.messages,
                )
            else:
                self.response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=1,
                    messages=self.messages,
                )

        elif self.model_key == 'gemini':
            self.response = self.client.generate_content(self.messages,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            })

        else:
            print('Incorrect Model name given please give correct model name')

        self.reset_count = 0

        # return the output of the model
        return self.response
    
    def reset_model(self):

        self.client = None

        if self.reset_count >= 3:
            return
        
        if self.model_key == 'gpt4o' or self.model_key == 'gpt4':
            file = open("OPENAI_API_KEY.txt", "r")
            api_key = file.read()
            self.client = OpenAI(api_key=api_key)

            if self.system_message is not None:
                system_prompt = {"role": "system", "content": [self.system_message]}
                self.messages.append(system_prompt)

        elif self.model_key == 'claude':
            file = open("ANTHROPIC_API_KEY.txt", "r")
            api_key = file.read()
            self.client = anthropic.Anthropic(api_key=api_key)
        
        elif self.model_key == 'gemini':
            file = open("GOOGLE_API_KEY.txt", "r")
            api_key = file.read()
            genai.configure(api_key=api_key)
            generation_config = genai.GenerationConfig(temperature=1)
            if self.system_message is not None:
                self.client = genai.GenerativeModel(model_name = self.model_name, system_instruction=self.system_message, generation_config=generation_config)
            else:
                self.client = genai.GenerativeModel(model_name = self.model_name, generation_config=generation_config)

        self.reset_count += 1

        print('Model is re-initiated...')

    def clean_model_output(self, output):
        # Remove any unescaped newline characters within the JSON string values
        cleaned_output = re.sub(r'(?<!\\)\n', ' ', output)
        
        # Replace curly quotes with straight quotes if necessary
        cleaned_output = cleaned_output.replace('“', '"').replace('”', '"')
        
        return cleaned_output

    def clean_response(self, response, path):
        # Correctly get the response from model
        if self.model_key == 'gpt4' or self.model_key == 'gpt4o':
            response_text = response.choices[0].message.content
        elif self.model_key == 'claude':
            response_text = response.content[0].text
        elif self.model_key == 'gemini':
            response_text = response.text
        
        if response_text == None:
            response_text = self.get_response()

        response_text = self.clean_model_output(response_text)

        # This regular expression finds the first { to the last }
        pattern = r'\{.*\}'
        # Search for the pattern
        match = re.search(pattern, response_text, flags=re.DOTALL)
        # Return the matched group which should be a valid JSON string
        if match != None:
            response_text = match.group(0)

        with open(path+'all_responses.txt', "a") as file:
            file.write(str(response_text) + '\n\n')

        try:
            response_text = json.loads(response_text)

        except json.JSONDecodeError as e:
            print('\n\nOutputError: received output that can\'t be read as json. Re-prompting the model.\n\n')

            # Create error message to reprompt the model
            error_message = 'Your output should be in a valid JSON format with a , after every key.'
            
            # Add the error message to the context
            self.add_user_message(user_msg=error_message)

            print('Generating new response...')
            while True:
                # See if you get the correct output
                try:
                    response = self.query_LLM()
                    print('\n\nProper response was generated')
                    break

                # If it doesn't then reset the model
                except:
                    print('Re-initiating model...')
                    self.reset_model()

                    if self.reset_count >= 3:
                        return None
            
                    
        
        return response_text
    
    def check_action(self, response_text):
        while True:
            # Check if the response is a dictionary
            if isinstance(response_text, dict):

                # Check if the key action exists 
                if "action" in response_text.keys():
                    
                    # Check to see if the action provided is correct and if so return the action
                    correct_output = self.env.action_space.n
                    if int(response_text["action"]) < correct_output:
                        return int(response_text["action"])
                    
                    else:
                        print('\n\nInvalid Action given (' + response_text['action']+') should be less than ' + correct_output + '.\n\n')
                        # Create error message to reprompt the model
                        error_message = 'You didn\'t give a valid action, make sure that it is a valid action from ' + str(0) + ' to ' + str(self.action_space-1) + ' for the action you would like to take.'
                        
                        # Add the error message to the context
                        self.add_user_message(user_msg=error_message)         
                    
                # If the action key is not in the json output then reprompt
                else:
                    print('\n\nInvalid JSON format given, no action key.\n\n')
                    # Create error message to reprompt the model
                    error_message = 'You didn\'t give a complete JSON output, it needs to include an \'action\' key which contains the numerical value for the action.'
                    
                    # Add the error message to the context
                    self.add_user_message(user_msg=error_message)
            
            # If the model response is not in the correct format then reprompt the model 
            else:
                print('\n\nInvalid response, not given as a JSON format.\n\n')
                # Create error message to reprompt the model
                error_message = 'You didn\'t give a valid response, it need to be a JSON format.'
                
                # Add the error message to the context
                self.add_user_message(user_msg=error_message)

            response = self.get_response()

            response_text = self.clean_response(response, self.path)

    def get_response(self):
        # Check to see if you get a response from the model
        try: 
            response = self.query_LLM()


        # If there is an error with generating a response (internal error)
        # Reset the model and try again
        except:
            print('\n\nReceived Error when generating response reseting model\n\n')
            
            # Reset model
            self.reset_model()

            while True:

                # See if you get the correct output
                try:
                    response = self.query_LLM()
                    print('\n\nReceived correct output continuing experiment.')
                    break
                # If it doesn't then reset the model
                except:
                    # Create error message to reprompt the model
                    error_message = 'Please provide a proper output'
                    
                    # Add the error message to the context
                    self.add_user_message(user_msg=error_message)

                    print('Re-initiating model...')
                    self.reset_model() 

                    # This means that more than likely you ran out of credits so break the code to not spend money
                    if self.reset_count >= 3:
                        return None
                    
        if response == 'idchoicescreatedmodelobjectsystem_fingerprintusage':
            response = self.get_response()
        
        return response


    def generate_response(self, path) -> str:   
        response = self.get_response()

        # Check if it is just reasoning or actual action output
        self.path = path

        response_text = self.clean_response(response, path)
        print('\n\nresponse: ', response_text)

        action_output = self.check_action(response_text)

        return action_output, response_text

    def add_user_message(self, frame=None, user_msg=None):
        if self.model_key == 'gpt4' or self.model_key == 'gpt4o':
            if user_msg is not None and frame is not None:
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_msg},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self.encode_image(frame)}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                )
            elif user_msg is not None and frame is None:
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_msg},
                        ],
                    }
                )
            elif user_msg is None and frame is not None:
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self.encode_image(frame)}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                )
            else:
                pass
        
        elif self.model_key == 'claude':
            if frame is not None and user_msg is not None:
                image_data = self.encode_image(frame)
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": user_msg
                            }
                        ]
                    }
                )
            elif frame is not None:
                image_data = self.encode_image(frame)
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data
                                }
                            }
                        ]
                    }
                )
            elif user_msg is not None:
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_msg
                            }
                        ]
                    }
                )

        if self.model_key == 'gemini':
            if frame is not None and user_msg is not None:
                image_data = self.encode_image(frame)
                self.messages.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "mime_type": "image/jpeg",
                                "data": image_data
                            },
                            {
                                "text": user_msg
                            }
                        ]
                    }
                )
            elif frame is not None and user_msg is None:
                image_data = self.encode_image(frame)
                self.messages.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "mime_type": "image/jpeg",
                                "data": image_data
                            }
                        ]
                    }
                )
            elif frame is None and user_msg is not None:
                self.messages.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": user_msg
                            }
                        ]
                    }
                )
            else:
                pass

    def add_assistant_message(self, demo_str=None):

        if self.model_key =='gpt4' or self.model_key =='gpt4o':
            if demo_str is not None:
                self.messages.append({"role": "assistant", "content": self.response})
                demo_str = None
                return
            
            if self.response is not None:
                self.messages.append({"role": "assistant", "content": self.response})
        
        elif self.model_key == 'claude':
            if demo_str is not None:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": demo_str},
                        ]
                    }
                )
                demo_str = None
                return

            if self.response is not None:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": self.response.content[0].text},
                        ]
                    }
                )

        elif self.model_key =='gemini':
            if demo_str is not None:
                self.messages.append(
                    {
                        "role": "model",
                        "parts": demo_str
                    }
                )
                demo_str = None
                return

            if self.response is not None:
                assistant_msg = self.response.text
                self.messages.append(
                    {
                        "role": "model",
                        "parts": assistant_msg
                    }
                )

        else:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ' '},
                    ]
                }
            )

    def delete_messages(self):
        print('Deleting Set of Messages...')

        if self.model_key == 'gpt4' or self.model_key == 'gpt4o':
            message_len = 9
        else:
            message_len = 8

        
        if len(self.messages) >= message_len:

            if self.messages[0]['role'] == 'system':
                # Delete user message
                value = self.messages.pop(1)

                # Delete Assistant message
                value = self.messages.pop(1)

            else:
                # Delete user message
                self.messages.pop(0)

                # Delete Assistant message
                self.messages.pop(0)