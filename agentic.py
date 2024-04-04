import os
import re
import subprocess
import logging
import sys
import tempfile
import pickle
from time import sleep
from dotenv import load_dotenv
from openai import OpenAI

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

logging.basicConfig(filename='agentic_workflow.log', level=logging.ERROR)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)
    return os.path.exists(filepath)

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
    return code_blocks[0].strip() if code_blocks else None

def test_code(code):
    if not code:
        return None, None

    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, 'temp_script.py')
        with open(script_path, 'w') as f:
            f.write(code)

        try:
            output = subprocess.check_output(['python', script_path], universal_newlines=True, stderr=subprocess.STDOUT, timeout=10)
            return output, None
        except subprocess.CalledProcessError as e:
            return None, e.output
        except subprocess.TimeoutExpired:
            return None, "Execution timed out after 10 seconds"
        except Exception as e:
            return None, str(e)

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

def get_code_suggestions(code, agent_role):
    suggestions = ""
    if agent_role == "Mike":
        suggestions = "Suggestions for Mike:\n"
        suggestions += "- ask Bob for help\n"
        suggestions += "- Consider using more descriptive variable names to improve code readability.\n"
        suggestions += "- Break down complex functions into smaller, reusable functions.\n"
        suggestions += "- Add comments to explain the purpose and functionality of key code segments.\n"
    elif agent_role == "Annie":
        suggestions = "Suggestions for Annie:\n"
        suggestions += "- Ask bob for help\n"
        suggestions += "- Review the code for potential logic errors and edge cases.\n"
        suggestions += "- Implement proper error handling and provide informative error messages to users.\n"
        suggestions += "- Ensure the user interface is intuitive and easy to navigate.\n"
        suggestions += "- Consider adding input validation to prevent unexpected behavior.\n"
    elif agent_role == "Alex":

        suggestions = "Suggestions for Alex:\n"
        suggestions += "- Ask Bob for help\n"
        suggestions += "- Send full robust script code ALWAYS free of placeholders. \n"
        suggestions += "- Implement a robust logging mechanism to track system activities and errors.\n"
        suggestions += "- Ensure the code follows best practices for security and compliance.\n"
        suggestions += "- Optimize the code for performance and efficiency.\n"
        suggestions += "- Develop comprehensive unit tests to cover critical functionalities.\n"
        suggestions += "- Automate the testing and deployment process to reduce manual effort.\n"
        suggestions += "- Set up continuous integration and continuous deployment (CI/CD) pipelines.\n"
    return suggestions

def perform_file_operation(operation, filepath, content=None):
    if operation == "read":
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                file_content = file.read()
            return f"File '{filepath}' read successfully. Contents:\n{file_content}"
        except FileNotFoundError:
            return f"File '{filepath}' not found."
    elif operation == "write":
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(content)
            return f"File '{filepath}' written successfully with content:\n{content}"
        except Exception as e:
            return f"Error writing to file '{filepath}': {str(e)}"
    elif operation == "append":
        try:
            with open(filepath, 'a', encoding='utf-8') as file:
                file.write(content)
            return f"Content appended to file '{filepath}' successfully. Appended content:\n{content}"
        except Exception as e:
            return f"Error appending to file '{filepath}': {str(e)}"
    else:
        return f"Invalid file operation: {operation}"

def main():
    max_iterations = 100
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
        mike_memory, annie_memory, bob_memory, alex_memory, code = [], [], [], [], ""

    print("\n" + "="*40 + " Conversation Start " + "="*40 + "\n")

    for i in range(1, max_iterations + 1):
        print("\n" + "-"*30 + f" Iteration {i} " + "-"*30 + "\n")

        # Bob's turn
        bob_input = f"You are Bob, the boss of Mike, Annie, and Alex. Here is the current state of the project:\n\nCurrent code: {code}\nCurrent error: {current_error if 'current_error' in locals() else 'None'}\n\nPlease provide your input as Bob."
        bob_response = agent_chat(bob_input, bob_system_message, bob_memory, "mixtral-8x7b-32768", 0.5)
        print(create_agent_response("Bob", bob_response, NEON_GREEN))

        # Mike's turn
        mike_input = f"You are Mike, an AI software architect and engineer. Here is the current state of the project:\n\nBob's message: {bob_response}\nCurrent code: {code}\nCurrent error: {current_error if 'current_error' in locals() else 'None'}\nSuggestions: {get_code_suggestions(code, 'Mike')}\n\nPlease provide your input as Mike."
        mike_response = agent_chat(mike_input, mike_system_message, mike_memory, "mixtral-8x7b-32768", 0.7)
        print(create_agent_response("Mike", mike_response, CYAN))
        code = open_file(current_code_file)
        current_error = test_code(code)

        # Annie's turn
        annie_input = f"You are Annie, a senior agentic workflow developer. Here is the current state of the project:\n\nBob's message: {bob_response}\nMike's response: {mike_response}\nCurrent code: {code}\nCurrent error: {current_error if 'current_error' in locals() else 'None'}\nSuggestions: {get_code_suggestions(code, 'Annie')}\n\nPlease provide your input as Annie."
        annie_response = agent_chat(annie_input, annie_system_message, annie_memory, "mixtral-8x7b-32768", 0.7)
        print(create_agent_response("Annie", annie_response, YELLOW))
        code = open_file(current_code_file)
        current_error = test_code(code)

        # Alex's turn
        alex_input = f"You are Alex, a DevOps Engineer. Here is the current state of the project:\n\nBob's message: {bob_response}\nMike's response: {mike_response}\nAnnie's response: {annie_response}\nCurrent code: {code}\nCurrent error: {current_error if 'current_error' in locals() else 'None'}\nSuggestions: {get_code_suggestions(code, 'Alex')}\n\nPlease provide your input as Alex."
        alex_response = agent_chat(alex_input, alex_system_message, alex_memory, "mixtral-8x7b-32768", 0.7)
        print(create_agent_response("Alex", alex_response, BLUE))
        code = open_file(current_code_file)
        current_error = test_code(code)

        if code:
            save_result = perform_file_operation("write", current_code_file, code)
            code = open_file(current_code_file)
            print(f"\n[FILE OPERATION] {save_result}")

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

        # Perform testing, quality checks, dependency updates, refactoring, optimization, and deployment as needed
        if current_code_error is None and current_code_output is not None:
            print("\n" + "-"*20 + " Running Tests " + "-"*20 + "\n")
            test_command = "python -m unittest discover tests"
            test_output, test_error = execute_terminal_command(test_command)
            if test_error:
                print(f"\n[ERROR] Tests encountered an error: {test_error}")
            else:
                print(f"\n[TESTS] Test output:\n{test_output}")

            print("\n" + "-"*20 + " Code Quality Checks " + "-"*20 + "\n")
            quality_check_command = "pylint current_code.py"
            quality_output, quality_error = execute_terminal_command(quality_check_command)
            if quality_error:
                print(f"\n[ERROR] Code quality checks encountered an error: {quality_error}")
            else:
                print(f"\n[QUALITY] Code quality output:\n{quality_output}")

            print("\n" + "-"*20 + " Updating Dependencies " + "-"*20 + "\n")
            update_command = "pip install --upgrade -r requirements.txt"
            update_output, update_error = execute_terminal_command(update_command)
            if update_error:
                print(f"\n[ERROR] Dependency update encountered an error: {update_error}")
            else:
                print(f"\n[UPDATE] Dependency update output:\n{update_output}")

            print("\n" + "-"*20 + " Code Refactoring " + "-"*20 + "\n")
            refactoring_command = "black current_code.py"
            refactoring_output, refactoring_error = execute_terminal_command(refactoring_command)
            if refactoring_error:
                print(f"\n[ERROR] Code refactoring encountered an error: {refactoring_error}")
            else:
                print(f"\n[REFACTOR] Code refactoring output:\n{refactoring_output}")

            print("\n" + "-"*20 + " Code Optimization " + "-"*20 + "\n")
            optimization_command = "python -m compileall current_code.py"
            optimization_output, optimization_error = execute_terminal_command(optimization_command)
            if optimization_error:
                print(f"\n[ERROR] Code optimization encountered an error: {optimization_error}")
            else:
                print(f"\n[OPTIMIZE] Code optimization output:\n{optimization_output}")

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
