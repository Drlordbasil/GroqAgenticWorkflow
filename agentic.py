import os
from agent_functions import agent_chat, extract_code, extract_task, generate_file_name, get_current_date_and_time, load_checkpoint, print_block, save_checkpoint
from code_execution_manager import CodeExecutionManager, run_tests, monitor_performance, optimize_code, format_code, pass_code_to_alex, send_status_update, generate_documentation

from crypto_wallet import CryptoWallet
from browser_tools import BrowserTools


def main():
    max_iterations = 500
    checkpoint_file = "agentic_workflow_checkpoint.pkl"
    code_execution_manager = CodeExecutionManager()
    date_time = get_current_date_and_time()
    def read_file(filepath):
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            print(f"Read file: {filepath}")
            return content
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None
        except Exception as e:
            print(f"Error reading file: {filepath}")
            return None
    browser_tools = BrowserTools()

    system_messages = {
        "mike": read_file("system_messages/mike.txt"),
        "annie": read_file("system_messages/annie.txt"),
        "bob": read_file("system_messages/bob.txt"),
        "alex": read_file("system_messages/alex.txt")
    }

    checkpoint_data, code = load_checkpoint(checkpoint_file)
    if checkpoint_data:
        memory = {key: value for key, value in zip(["mike", "annie", "bob", "alex"], checkpoint_data)}
    else:
        memory = {key: [] for key in ["mike", "annie", "bob", "alex"]}

    print_block("Agentic Workflow", character='*')
    print_block(f"Start Time: {date_time}")

    for i in range(1, max_iterations + 1):
        print_block(f"Iteration {i}")
        files = code_execution_manager.list_files_in_workspace()
        project_output_goal = f"Create a profitable script from scratch and run it in the workspace. Ensure the script generates real profit, not simulated profit. Follow software engineering best practices, including reflection, refactoring, and step-by-step guidelines. Use available tools for research and information gathering as needed. Current time: {date_time}"

        for agent in ["mike", "annie", "bob", "alex"]:
            memory[agent].append({"role": "system", "content": f"Files in workspace: {files}"})
            memory[agent].append({"role": "system", "content": f"Iteration {i} started. Current time: {date_time}"})

        bob_input = f"[Python experts only, ensure high-quality code] Current time: {date_time}. You are Bob (money-minded micromanager), the boss of Mike, Annie, and Alex. Guide the team in creating a profitable script from scratch that generates real profit, not simulated profit. Ensure that the team follows software engineering best practices, including reflection, refactoring, and step-by-step guidelines. Encourage the team to create robust, verbose, non-pseudo, and non-example final code implementations for real-world cases. Remind them to use available tools for research and information gathering as needed.\nHere is the current state of the project:\n\nProject Goal: {project_output_goal}\nCurrent files in the workspace: {files}\n\nPlease provide your input as Bob, including delegating tasks to Mike, Annie, and Alex based on their expertise and the project requirements. Encourage the team to brainstorm ideas, utilize available tools for research, and collaborate effectively to create a script that meets the project's goals."

        bob_response = agent_chat(bob_input, system_messages["bob"], memory["bob"], "mixtral-8x7b-32768", 0.5, agent_name="Bob")
        print(f"Bob's Response:\n{bob_response}")

        for agent in ["mike", "annie", "alex"]:
            agent_input = f"Current time: {date_time}. You are {agent.capitalize()}, an AI {'software architect and engineer' if agent == 'mike' else 'senior agentic workflow developer' if agent == 'annie' else 'DevOps Engineer'}. Here is your task from Bob:\n\nTask: {extract_task(bob_response, agent.capitalize())}\n\nCurrent files in the workspace: {files}\n\nPlease provide your response, including any ideas, code snippets, or suggestions for creating a profitable script from scratch that generates real profit. Focus on creating high-quality, efficient, and well-documented code that follows software engineering best practices, including reflection and refactoring. Utilize available tools for research and information gathering as needed. Collaborate with your teammates to ensure a cohesive and functional script. Provide robust, verbose, non-pseudo, and non-example final code implementations for real-world cases."

            agent_response = agent_chat(agent_input, system_messages[agent], memory[agent], "mixtral-8x7b-32768", 0.7, agent_name=agent.capitalize())
            print(f"{agent.capitalize()}'s Response:\n{agent_response}")

            # Extract code from agent's response
            agent_code = extract_code(agent_response)
            if agent_code:
                # Pass code to Alex for review and deployment
                pass_code_to_alex(agent_code, memory["alex"])

        # Check if Alex provided any code updates
        alex_code = extract_code(agent_response)
        if alex_code:
            code = alex_code
            file_name = generate_file_name(code)
            code_execution_manager.save_file(file_name, code)
            memory["alex"].append({"role": "system", "content": f"Code saved in workspace as {file_name}."})

            print_block("Verifying Real Profit Generation")
            profit_verification_input = f"Please verify that the current code generates real profit and not simulated profit. Provide evidence and explanations to support your verification.\n\nCurrent code:\n\n{code}"
            profit_verification_response = agent_chat(profit_verification_input, system_messages["bob"], memory["bob"], "mixtral-8x7b-32768", 0.5, agent_name="Bob")
            print(f"Bob's Profit Verification:\n{profit_verification_response}")

            send_status_update(memory["mike"], memory["annie"], memory["alex"], f"Iteration {i} completed. Code created from scratch and verified for real profit generation. Code saved under file name: {file_name} in workspace. Code for {file_name} is:\n\n{code}")

        checkpoint_data = [memory[key] for key in ["mike", "annie", "bob", "alex"]] + [code]
        save_checkpoint(checkpoint_data, checkpoint_file, code)

    print_block("Agentic Workflow Completed", character='*')
    browser_tools.close()  # Close the browser

if __name__ == "__main__":
    main()