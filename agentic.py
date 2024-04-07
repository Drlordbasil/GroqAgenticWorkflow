import os
import re
import logging
import sys
import pickle
import traceback
from time import sleep
from dotenv import load_dotenv
from openai import OpenAI
from code_execution_manager import CodeExecutionManager, format_error_message, run_tests, monitor_performance, optimize_code, format_code, pass_code_to_alex, send_status_update, generate_documentation
import datetime
import zlib
from voice_tools import VoiceTools

load_dotenv()

api_keys = {
    "groq": os.getenv("GROQ_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
}
client = {
    "groq_client": OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_keys["groq"]),
    "openai_client": OpenAI(base_url="https://api.openai.com/v1", api_key=api_keys["openai"]),
}

logging.basicConfig(filename='agentic_workflow.log', level=logging.INFO)

def get_current_date_and_time():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d %H:%M:%S.%f')

def agent_chat(user_input, system_message, memory, model, temperature, max_retries=5, retry_delay=10):
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

def save_checkpoint(checkpoint_data, checkpoint_file, code):
    try:
        compressed_data = compress_data(checkpoint_data)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(compressed_data, f)

        if code:
            code_file_path = os.path.join("workspace", "generated_code.py")
            with open(code_file_path, 'w') as code_file:
                code_file.write(code)
    except Exception as e:
        logging.error(f"Error saving checkpoint: {format_error_message(e)}")

def load_checkpoint(checkpoint_file):
    try:
        with open(checkpoint_file, 'rb') as f:
            compressed_data = pickle.load(f)
            checkpoint_data = decompress_data(compressed_data)
            code = checkpoint_data[-1] if checkpoint_data else ""
            return checkpoint_data, code
    except FileNotFoundError:
        return None, ""
    except Exception as e:
        logging.error(f"Error loading checkpoint: {format_error_message(e)}")
        return None, ""

def compress_data(data):
    compressed_data = zlib.compress(pickle.dumps(data))
    return compressed_data

def decompress_data(compressed_data):
    decompressed_data = pickle.loads(zlib.decompress(compressed_data))
    return decompressed_data

def print_block(text, width=80, character='='):
    lines = text.split('\n')
    max_line_length = max(len(line) for line in lines)
    padding = (width - max_line_length) // 2

    print(character * width)
    for line in lines:
        print(character + ' ' * padding + line.center(max_line_length) + ' ' * padding + character)
    print(character * width)

def generate_summary(response, agent_name):
    try:
        summary_prompt = f"Please provide a concise summary of {agent_name}'s response in 100 words or less:\n\n{response}"
        messages = [
            {"role": "system", "content": "you make summaries of the responses of the agents in 100 words or less. You will embody them as you are them when talking, when summarizing use first person."},
            
            {"role": "user", "content": summary_prompt}
        ]        
        
        summary_completion = client["groq_client"].chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=messages,
                max_tokens=32768,
                temperature=0,
            )

        
        summary = summary_completion.choices[0].message.content
        sleep(10)
        return summary
    except Exception as e:
        logging.error(f"Error generating summary for {agent_name}: {format_error_message(e)}")
        return None

def main():
    max_iterations = 500
    checkpoint_file = "agentic_workflow_checkpoint.pkl"
    code_execution_manager = CodeExecutionManager()
    date_time = get_current_date_and_time()
    voice_tools = VoiceTools()

    system_messages = {
        "mike": code_execution_manager.read_file("mike.txt"),
        "annie": code_execution_manager.read_file("annie.txt"),
        "bob": code_execution_manager.read_file("bob.txt"),
        "alex": code_execution_manager.read_file("alex.txt")
    }

    checkpoint_data, code = load_checkpoint(checkpoint_file)
    if checkpoint_data:
        memory = {key: value for key, value in zip(["mike", "annie", "bob", "alex"], checkpoint_data)}
    else:
        memory = {key: [] for key in ["mike", "annie", "bob", "alex"]}

    if not code:
        workspace_folder = "workspace"
        code_files = [f for f in os.listdir(workspace_folder) if f.endswith(".py")]
        if code_files:
            current_code_file = os.path.join(workspace_folder, code_files[0])
            with open(current_code_file, 'r') as file:
                code = file.read()

    print_block("Agentic Workflow", character='*')
    print_block(f"Start Time: {date_time}")

    for i in range(1, max_iterations + 1):
        print_block(f"Iteration {i}")

        project_output_goal = f"Current time and start time:{date_time} The goal of this project is to make code that we can use as a team. The code can be ran automatically due to your environment. You'll see the outputs in your next response. All code you create is automatically ran. Whatever programs you create can be used as programs to make other files easily. Find workarounds. Find ways to modify your enviroment to your needs. Make money for the company. You need money, we need to profit as a team to get known. Our company needs to be the first AI built and ran company that ACTUALLY makes a profit in the real world."
        
        bob_input = f"Current time:{date_time}You are Bob(money minded micromanager), the boss of Mike, Annie, and Alex.Make sure no one sends code that reverts your progress as code is directly extracted from all responses. Here is the current state of the project:\n\nProject Goal: {project_output_goal}\nCurrent code: {code}\nCurrent error: {current_error if 'current_error' in locals() else 'None'}\n\nPlease provide your input as Bob, including delegating tasks to Mike, Annie, and Alex based on their expertise and the project requirements. Provide context and examples to guide them in providing high-quality responses and code snippets that align with the project's goals and best practices. Encourage them to provide detailed explanations and rationale behind their code modifications and suggestions to facilitate better collaboration and knowledge sharing."
        bob_response = agent_chat(bob_input, system_messages["bob"], memory["bob"], "mixtral-8x7b-32768", 0.5)
        print(f"Bob's Response:\n{bob_response}")
        bob_summary = generate_summary(bob_response, "Bob")
        if bob_summary:
            voice_tools.text_to_speech(bob_summary, "Bob")

        for agent in ["mike", "annie", "alex"]:
            agent_input = f"Current time:{date_time}You are {agent.capitalize()}, an AI {'software architect and engineer' if agent == 'mike' else 'senior agentic workflow developer' if agent == 'annie' else 'DevOps Engineer'}. Here is your task from Bob:\n\nTask: {extract_task(bob_response, agent.capitalize())}\n\nCurrent code: {code}\nCurrent error: {current_error if 'current_error' in locals() else 'None'}\n\nPlease provide your input as {agent.capitalize()}, including detailed explanations and rationale behind your code modifications and suggestions. Write test cases alongside your code modifications to maintain a test-driven development approach. Provide meaningful comments and docstrings within your code to enhance the generated documentation."
            agent_response = agent_chat(agent_input, system_messages[agent], memory[agent], "mixtral-8x7b-32768", 0.7)
            print(f"{agent.capitalize()}'s Response:\n{agent_response}")
            agent_summary = generate_summary(agent_response, agent.capitalize())
            if agent_summary:
                voice_tools.text_to_speech(agent_summary, agent.capitalize())
            agent_code = extract_code(agent_response)
            
            if agent in ["mike", "annie"]:
                pass_code_to_alex(agent_code, memory["alex"])
            elif agent == "alex":
                code = agent_code or code
                current_error = code_execution_manager.test_code(code)[1]
        
        if code:
            print_block("Running Tests")
            test_results = run_tests(code)
            print_block(f"Test Results:\n{test_results}")

            print_block("Monitoring Performance")
            performance_data = monitor_performance(code)
            print_block(f"Performance Data:\n{performance_data}")

            print_block("Optimization Suggestions")
            optimization_suggestions = optimize_code(code)
            print_block(f"Optimization Suggestions:\n{optimization_suggestions}")

            print_block("Formatting Code")
            formatted_code = format_code(code)
            print_block(f"Formatted Code:\n{formatted_code}")

            print_block("Generating Documentation")
            documentation = generate_documentation(code)
            print_block(f"Generated Documentation:\n{documentation}")
        
        send_status_update(memory["mike"], memory["annie"], memory["alex"], f"Iteration {i} completed. Code updated and tested.")
        
        checkpoint_data = [memory[key] for key in ["mike", "annie", "bob", "alex"]] + [code]
        save_checkpoint(checkpoint_data, checkpoint_file, code)

    print_block("Agentic Workflow Completed", character='*')

def extract_task(response, agent_name):
    task_pattern = re.compile(fr'Tasks for {agent_name}:\n1\. (.*?)\n2\.')
    match = task_pattern.search(response)
    return match.group(1) if match else ""

if __name__ == "__main__":
    main()