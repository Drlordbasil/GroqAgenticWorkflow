from dotenv import load_dotenv
import os
import requests
from openai import OpenAI

from time import sleep
import sys

import re
import subprocess
import logging
import tempfile

import pickle


load_dotenv()

api_keys = {
   "groq": os.getenv("GROQ_API_KEY"),
   
   "openai": os.getenv("OPENAI_API_KEY"),
}
client = {
   "groq_client": OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_keys["groq"]),
   "openai_client": OpenAI(base_url="https://api.openai.com/v1", api_key=api_keys["openai"]),
}

voiceid1 = "21m00Tcm4TlvDq8ikWAM"
voiceid2 = "29vD33N1CtxCmqQRPOH"

NEON_GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET_COLOR = "\033[0m"

logging.basicConfig(filename='agentic_workflow.log', level=logging.ERROR)

def open_file(filepath):
   with open(filepath, 'r', encoding='utf-8') as infile:
       return infile.read()

def save_file(filepath, content):
   with open(filepath, 'w', encoding='utf-8') as outfile:
       outfile.write(content)
   if os.path.exists(filepath):
       return True
   else:
       return False





def agent_chat(user_input, system_message, memory, model, temperature, max_retries=30, retry_delay=60):
    messages = [
        {"role": "system", "content": system_message},
        *memory[-5:],
        {"role": "user", "content": user_input}
    ]

    retry_count = 0
    while retry_count < max_retries:
        try:
            chat_completion = client["groq_client"].chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=32768,
                temperature=temperature,
            )

            response_content = chat_completion.choices[0].message.content
            truncated_response = response_content[:1000]
            memory.append({"role": "system", "content": truncated_response})
            memory.append({"role": "user", "content": user_input})
            sleep(10)
            return response_content

        except Exception as e:
            logging.error(f"Error in agent_chat: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                logging.info(f"Retrying in {retry_delay} seconds... (Attempt {retry_count}/{max_retries})")
                sleep(retry_delay)
            else:
                logging.error(f"Max retries exceeded. Raising the exception.")
                raise e


def extract_code(text):
   code_block_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
   code_blocks = code_block_pattern.findall(text)
   if code_blocks:
       return code_blocks[0].strip()
   else:
       return None

def test_code(code):
   if code is None:
       return None, None

   with tempfile.TemporaryDirectory() as temp_dir:
       script_path = os.path.join(temp_dir, 'temp_script.py')
       with open(script_path, 'w') as f:
           f.write(code)

       try:
           output = subprocess.check_output(['python', script_path], universal_newlines=True, stderr=subprocess.STDOUT, timeout=10)
           return output, None

       except subprocess.CalledProcessError as e:
           error_message = e.output
           return None, error_message

       except subprocess.TimeoutExpired as e:
           error_message = "Execution timed out after 10 seconds"
           return None, error_message

       except Exception as e:
           error_message = str(e)
           return None, error_message

def execute_terminal_command(command):
   try:
       result = subprocess.run(command=f"cd workspace\n {command}", capture_output=True, text=True, shell=True)
       return result.stdout, result.stderr
   except Exception as e:
       logging.error(f"Error executing terminal command: {str(e)}")
       
       return None, str(e)

def create_agent_response(agent_name, agent_response, agent_color):
   return agent_color + f"\n{agent_name}: " + agent_response + RESET_COLOR

def save_checkpoint(checkpoint_data, checkpoint_file):
   with open(checkpoint_file, 'wb') as f:
       pickle.dump(checkpoint_data, f)

def load_checkpoint(checkpoint_file):
   try:
       with open(checkpoint_file, 'rb') as f:
           return pickle.load(f)
   except FileNotFoundError:
       return None
def main():

    max_range = 100
    
    checkpoint_file = "agentic_workflow_checkpoint.pkl"
    current_code_file = "current_code.py"

    mike_system_message = open_file("mike.txt")
    annie_system_message = open_file("annie.txt")
    bob_system_message = open_file("bob.txt")
    alex_system_message = open_file("alex.txt")

    checkpoint_data = load_checkpoint(checkpoint_file)
    if checkpoint_data:
        mike_memory, annie_memory, bob_memory, alex_memory, code = checkpoint_data
    else:
        mike_memory = []
        annie_memory = []
        bob_memory = []
        alex_memory = []

    print("\n" + "="*40 + " Conversation Start " + "="*40 + "\n")

    user_input = """
    Hello Annie, Mike, and Alex. I am your boss Bob. Let's work on creating the first agentic automated entrepreneurial profit engineering agent, we need to keep code in our convo at all times. Let's start with building the frame of the 1 file script.

    any code you markdown will be saved as the final code, updating it whenever I detect a new code block. Let's start with the first code block and just build on it from there slowly and robustly. We can simply get the errors the next response to us, as this is fully automated when code is detected, but alex will also be able to use local commands, like pip install openai or pip install requests, etc. He can also make sure to create, list, save, ect to the workspace folder within your current working directory. Let's start with the first code block and just build on it from there slowly and robustly. We can simply get the errors the next response to us, as this is fully automated when code is detected, but alex will also be able to use local commands, like pip install openai or pip install requests, etc. He can also make sure to create, list, save, ect to the workspace folder within your current working directory.
    
    Remember to understand that we must use opensource commercially available models from huggingface or create our own models for our use-cases depending, we should never make code that requires an API key at all.
    """

    bob_response = user_input

    for i in range(1, max_range):
        print("\n" + "-"*30 + f" Iteration {i} " + "-"*30 + "\n")

        # Mike's turn
        mike_input = f"Remember to understand that we must use opensource commercially available models from huggingface or create our own models for our use-cases depending, we should never make code that requires an API key at all.You are Mike, an AI software architect and engineer. Here is the current state of the project:\n\nBob's message: {bob_response}\nCurrent code: {code}\nCurrent error: {mike_error if 'mike_error' in locals() else 'None'}\n\nPlease provide your input as Mike."
        mike_response = agent_chat(mike_input, mike_system_message, mike_memory, "mixtral-8x7b-32768", 0.7)
        print(create_agent_response("Mike", mike_response, CYAN))
        code = extract_code(mike_response)
        mike_output, mike_error = test_code(code)

        # Annie's turn
        annie_input = f"Remember to understand that we must use opensource commercially available models from huggingface or create our own models for our use-cases depending, we should never make code that requires an API key at all.You are Annie, a senior agentic workflow developer. Here is the current state of the project:\n\nMike's response: {mike_response}\nCurrent code: {code}\nCurrent error: {annie_error if 'annie_error' in locals() else 'None'}\n\nPlease provide your input as Annie."
        annie_response = agent_chat(annie_input, annie_system_message, annie_memory, "mixtral-8x7b-32768", 0.7)
        print(create_agent_response("Annie", annie_response, YELLOW))
        code = extract_code(annie_response)
        annie_output, annie_error = test_code(code)

        # Alex's turn
        alex_input = f"Remember to understand that we must use opensource commercially available models from huggingface or create our own models for our use-cases depending, we should never make code that requires an API key at all.You are Alex, a DevOps Engineer. Here is the current state of the project:\n\nMike's response: {mike_response}\nAnnie's response: {annie_response}\nCurrent code: {code}\nCurrent error: {alex_error if 'alex_error' in locals() else 'None'}\n\nPlease provide your input as Alex."
        alex_response = agent_chat(alex_input, alex_system_message, alex_memory, "mixtral-8x7b-32768", 0.7)
        print(create_agent_response("Alex", alex_response, BLUE))
        code = extract_code(alex_response)
        alex_output, alex_error = test_code(code)

        # Bob's turn
        bob_input = f"Remember to understand that we must use opensource commercially available models from huggingface or create our own models for our use-cases depending, we should never make code that requires an API key at all.You are Bob, the boss of Mike, Annie, and Alex. Here are the messages from your employees:\n\nMike's response: {mike_response}\nAnnie's response: {annie_response}\nAlex's response: {alex_response}\nCurrent code: {code}\nCurrent error: {bob_error if 'bob_error' in locals() else 'None'}\n\nPlease provide your input as Bob."
        bob_response = agent_chat(bob_input, bob_system_message, bob_memory, "mixtral-8x7b-32768", 0.5)
        print(create_agent_response("Bob", bob_response, NEON_GREEN))
        code = extract_code(bob_response)
        bob_output, bob_error = test_code(code)

        if code:
            save_file(current_code_file, code)

        checkpoint_data = (mike_memory, annie_memory, bob_memory, alex_memory, code)
        save_checkpoint(checkpoint_data, checkpoint_file)

        print("\n" + "-"*20 + " Current Code Execution " + "-"*20 + "\n")
        current_code_output, current_code_error = test_code(code)
        if current_code_error:
            print(f"\n[ERROR] Current code encountered an error: {current_code_error}")
        elif current_code_output:
            print(f"\n[OUTPUT] Current code output:\n{current_code_output}")
        else:
            print("\nNo output from the current code.")

        if i % 5 == 0:
            print("\n" + "-"*20 + " Running Tests " + "-"*20 + "\n")
            test_command = "python -m unittest discover tests"
            test_output, test_error = execute_terminal_command(test_command)
            if test_output:
                print(f"\n[TESTS] Tests output:\n{test_output}")
            if code:
                save_file(current_code_file, code)

            if test_error:
                print(f"\n[ERROR] Tests encountered an error: {test_error}")
            else:
                print(f"\n[TESTS] Test output:\n{test_output}")

        if i % 10 == 0:
            print("\n" + "-"*20 + " Code Quality Checks " + "-"*20 + "\n")
            quality_check_command = "pylint current_code.py"
            quality_output, quality_error = execute_terminal_command(quality_check_command)
            if quality_error:
                print(f"\n[ERROR] Code quality checks encountered an error: {quality_error}")
            else:
                print(f"\n[QUALITY] Code quality output:\n{quality_output}")

        if i % 15 == 0:
            print("\n" + "-"*20 + " Updating Dependencies " + "-"*20 + "\n")
            update_command = "pip install --upgrade -r requirements.txt"
            update_output, update_error = execute_terminal_command(update_command)
            if update_error:
                print(f"\n[ERROR] Dependency update encountered an error: {update_error}")
            else:
                print(f"\n[UPDATE] Dependency update output:\n{update_output}")

        if i % 20 == 0:
            print("\n" + "-"*20 + " Code Refactoring " + "-"*20 + "\n")
            refactoring_command = "black current_code.py"
            refactoring_output, refactoring_error = execute_terminal_command(refactoring_command)
            if refactoring_error:
                print(f"\n[ERROR] Code refactoring encountered an error: {refactoring_error}")
            else:
                print(f"\n[REFACTOR] Code refactoring output:\n{refactoring_output}")

        if i % 25 == 0:
            print("\n" + "-"*20 + " Code Optimization " + "-"*20 + "\n")
            optimization_command = "python -m compileall current_code.py"
            optimization_output, optimization_error = execute_terminal_command(optimization_command)
            if optimization_error:
                print(f"\n[ERROR] Code optimization encountered an error: {optimization_error}")
            else:
                print(f"\n[OPTIMIZE] Code optimization output:\n{optimization_output}")

        if i % 30 == 0:
            save_file(current_code_file, code)
            print("\n" + "-"*20 + " Code Deployment " + "-"*20 + "\n")
            deployment_command = "python current_code.py"
            deployment_output, deployment_error = execute_terminal_command(deployment_command)
            if deployment_error:
                print(f"\n[ERROR] Code deployment encountered an error: {deployment_error}")
            else:
                print(f"\n[DEPLOY] Code deployment output:\n{deployment_output}")

    print("\n" + "="*40 + " Conversation End " + "="*40 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt detected. Exiting the program.")
        sys.exit(0)
