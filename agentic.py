import os
import re
import logging
import sys
import pickle
import traceback
from time import sleep
from dotenv import load_dotenv
from openai import OpenAI
from code_execution_manager import CodeExecutionManager, format_error_message, run_tests, monitor_performance, optimize_code, pass_code_to_alex, send_status_update, generate_documentation, commit_changes
import datetime

load_dotenv()

api_keys = {
    "groq": os.getenv("GROQ_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
}
client = {
    "groq_client": OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_keys["groq"]),
    "openai_client": OpenAI(base_url="https://api.openai.com/v1", api_key=api_keys["openai"]),
}

NEON_GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET_COLOR = "\033[0m"

logging.basicConfig(filename='agentic_workflow.log', level=logging.INFO)
def get_current_date_and_time():
    """
    Returns the current date and time as a string in the format
    'YYYY-MM-DD HH:MM:SS.ssssss'
    """
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d %H:%M:%S.%f')
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
            logging.error(f"Error in agent_chat: {format_error_message(e)}")
            retry_count += 1
            if retry_count < max_retries:
                logging.info(f"Retrying in {retry_delay} seconds... (Attempt {retry_count}/{max_retries})")
                sleep(retry_delay)
            else:
                logging.error(f"Max retries exceeded. Raising the exception.")
                raise e

def extract_code(text):
    try:
        code_block_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
        code_blocks = code_block_pattern.findall(text)
        return code_blocks[0].strip() if code_blocks else None
    except Exception as e:
        logging.error(f"Error extracting code: {format_error_message(e)}")
        return None

def create_agent_response(agent_name, agent_response, agent_color):
    return agent_color + f"\n{agent_name}: " + agent_response + RESET_COLOR

def save_checkpoint(checkpoint_data, checkpoint_file, code):
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    # Save the code to a file in the workspace folder only if it's not None
    if code:
        code_file_path = os.path.join("workspace", "generated_code.py")
        with open(code_file_path, 'w') as code_file:
            code_file.write(code)

def load_checkpoint(checkpoint_file):
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            code = checkpoint_data[-1] if checkpoint_data else ""
            return checkpoint_data, code
    except FileNotFoundError:
        return None, ""

def extract_task(response, agent_name):
    task_pattern = re.compile(fr'Tasks for {agent_name}:\n1\. (.*?)\n2\.')
    match = task_pattern.search(response)
    return match.group(1) if match else ""

def bob_delegate_tasks(bob_response, mike_memory, annie_memory, alex_memory):
    mike_task = extract_task(bob_response, "Mike")
    annie_task = extract_task(bob_response, "Annie")
    alex_task = extract_task(bob_response, "Alex")
    
    mike_memory.append({"role": "system", "content": f"Task from Bob: {mike_task}"})
    annie_memory.append({"role": "system", "content": f"Task from Bob: {annie_task}"})
    alex_memory.append({"role": "system", "content": f"Task from Bob: {alex_task}"})
    
    return mike_task, annie_task, alex_task

def main():
    max_iterations = 100
    checkpoint_file = "agentic_workflow_checkpoint.pkl"
    current_code_file = "current_code.py"
    code_execution_manager = CodeExecutionManager()
    date_time = get_current_date_and_time()
    
    mike_system_message = code_execution_manager.read_file("mike.txt")
    annie_system_message = code_execution_manager.read_file("annie.txt")
    bob_system_message = code_execution_manager.read_file("bob.txt")
    alex_system_message = code_execution_manager.read_file("alex.txt")

    checkpoint_data, code = load_checkpoint(checkpoint_file)
    if checkpoint_data:
        mike_memory, annie_memory, bob_memory, alex_memory, *_ = checkpoint_data
    else:
        mike_memory, annie_memory, bob_memory, alex_memory = [], [], [], []

    if not code:
        # Check the workspace folder for existing code files
        workspace_folder = "workspace"
        code_files = [f for f in os.listdir(workspace_folder) if f.endswith(".py")]
        if code_files:
            current_code_file = os.path.join(workspace_folder, code_files[0])
            with open(current_code_file, 'r') as file:
                code = file.read()

    print("\n" + "="*40 + " Conversation Start " + "="*40 + "\n")

    for i in range(1, max_iterations + 1):
        print("\n" + "-"*30 + f" Iteration {i} " + "-"*30 + "\n")
        project_output_goal = f"Current time and start time:{date_time} The goal of this project is to develop a robust AI-based workflow management system that can automate various tasks and streamline business processes. The system should be scalable, secure, and user-friendly, with a focus on efficiency and reliability. Use the latest AI technologies and best practices to create a cutting-edge solution that meets the needs of our clients and stakeholders. The project timeline is 2 hours, and the budget is $500. The success criteria include achieving a minimum accuracy of 95 percent and full code free of placeholder logic."
        
        bob_input = f"Current time:{date_time}You are Bob, the boss of Mike, Annie, and Alex. Here is the current state of the project:\n\nProject Goal: {project_output_goal}\nCurrent code: {code}\nCurrent error: {current_error if 'current_error' in locals() else 'None'}\n\nPlease provide your input as Bob, including delegating tasks to Mike, Annie, and Alex based on their expertise and the project requirements."
        bob_response = agent_chat(bob_input, bob_system_message, bob_memory, "mixtral-8x7b-32768", 0.5)
        print(create_agent_response("Bob", bob_response, NEON_GREEN))

        mike_task, annie_task, alex_task = bob_delegate_tasks(bob_response, mike_memory, annie_memory, alex_memory)

        mike_input = f"Current time:{date_time}You are Mike, an AI software architect and engineer.ALWAYS SEND PYTHON CODE FREE OF PLACEHOLDERS!!!! Here is your task from Bob:\n\nTask: {mike_task}\n\nCurrent code: {code}\nCurrent error: {current_error if 'current_error' in locals() else 'None'}\n\nPlease provide your input as Mike."
        mike_response = agent_chat(mike_input, mike_system_message, mike_memory, "mixtral-8x7b-32768", 0.7)
        print(create_agent_response("Mike", mike_response, CYAN))
        mike_code = extract_code(mike_response)
        current_error = code_execution_manager.test_code(code)
        pass_code_to_alex(mike_code, alex_memory)

        annie_input = f"Current time:{date_time}You are Annie, a senior agentic workflow developer.ALWAYS SEND PYTHON CODE FREE OF PLACEHOLDERS!!!! Here is your task from Bob:\n\nTask: {annie_task}\n\nCurrent code: {code}\nCurrent error: {current_error if 'current_error' in locals() else 'None'}\n\nPlease provide your input as Annie."
        annie_response = agent_chat(annie_input, annie_system_message, annie_memory, "mixtral-8x7b-32768", 0.7)
        print(create_agent_response("Annie", annie_response, YELLOW))
        annie_code = extract_code(annie_response)
        current_error = code_execution_manager.test_code(code)
        pass_code_to_alex(annie_code, alex_memory)

        alex_input = f"Current time:{date_time}You are Alex, a DevOps Engineer. ALWAYS SEND PYTHON CODE FREE OF PLACEHOLDERS!!!!Here is your task from Bob:\n\nTask: {alex_task}\n\nCurrent code: {code}\nCurrent error: {current_error if 'current_error' in locals() else 'None'}\n\nPlease provide your input as Alex."
        alex_response = agent_chat(alex_input, alex_system_message, alex_memory, "mixtral-8x7b-32768", 0.7)
        print(create_agent_response("Alex", alex_response, BLUE))
        alex_code = extract_code(alex_response)
        current_error = code_execution_manager.test_code(code)

        code = alex_code or code

        if code:
            run_tests(code)
            monitor_performance(code)
            optimize_code(code)
            commit_changes(code)
            generate_documentation(code)

        send_status_update(mike_memory, annie_memory, alex_memory, f"Iteration {i} completed. Code updated and tested.")

        checkpoint_data = (mike_memory, annie_memory, bob_memory, alex_memory, code)
        save_checkpoint(checkpoint_data, checkpoint_file, code)

    print("\n" + "="*40 + " Conversation End " + "="*40 + "\n")

if __name__ == "__main__":
    main()
