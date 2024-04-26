import os
from agent_functions import agent_chat, extract_code, get_current_date_and_time, load_checkpoint, print_block, save_checkpoint
from code_execution_manager import CodeExecutionManager
from browser_tools import BrowserTools

code_execution_manager = CodeExecutionManager()
browser_tools = BrowserTools()

def read_file(filepath):
    """
    Read the contents of a file.

    Args:
        filepath (str): The path to the file.

    Returns:
        str: The contents of the file, or None if an error occurred.
    """
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

def read_multiple_files(files):
    """
    Read the contents of multiple files.

    Args:
        files (list): A list of file paths.

    Returns:
        str: The concatenated contents of the files.
    """
    content = ""
    for file in files:
        file_content = code_execution_manager.read_file(file)
        if file_content and file_content.get("status") == "success":
            content += file_content.get("content", "") + "\n"
    return content

def pass_code_to_alex(code, alex_memory):
    """
    Pass the code to Alex for review.

    Args:
        code (str): The code to be reviewed.
        alex_memory (list): Alex's memory.

    Returns:
        None
    """
    alex_memory.append({"role": "system", "content": f"Code from Mike and Annie: {code}"})

def send_status_update(mike_memory, annie_memory, alex_memory, project_status):
    """
    Send a status update to Mike, Annie, and Alex.

    Args:
        mike_memory (list): Mike's memory.
        annie_memory (list): Annie's memory.
        alex_memory (list): Alex's memory.
        project_status (str): The project status update.

    Returns:
        None
    """
    mike_memory.append({"role": "system", "content": f"Project Status Update: {project_status}"})
    annie_memory.append({"role": "system", "content": f"Project Status Update: {project_status}"})
    alex_memory.append({"role": "system", "content": f"Project Status Update: {project_status}"})

def main():
    """
    The main function orchestrating the agentic workflow.
    """
    max_iterations = 500
    checkpoint_file = "agentic_workflow_checkpoint.pkl"

    system_messages = {
        "mike": read_file("system_messages/mike.txt"),
        "annie": read_file("system_messages/annie.txt"),
        "bob": read_file("system_messages/bob.txt"),
        "alex": read_file("system_messages/alex.txt")
    }

    checkpoint_data, code = load_checkpoint(checkpoint_file)
    memory = {key: value for key, value in zip(["mike", "annie", "bob", "alex"], checkpoint_data)} if checkpoint_data else {key: [] for key in ["mike", "annie", "bob", "alex"]}

    print_block("Agentic Workflow", character='*')
    date_time = get_current_date_and_time()
    print_block(f"Start Time: {date_time}")

    for i in range(1, max_iterations + 1):
        print_block(f"Iteration {i}")
        files = code_execution_manager.list_files_in_workspace()
        if files.get("status") == "success":
            workspace_files = files.get("files", [])
        else:
            workspace_files = []
        project_output_goal = f"Create a code/script making script/program with Python. This script will run and create a new script itself. Ensure that all Python scripts created work together properly without requiring API keys or secrets. Run the script in the workspace. The generated script should produce real profit, not simulated profit. Follow software engineering best practices, including reflection, refactoring, and step-by-step guidelines. Use available tools for research and information gathering as needed. Current time: {date_time}"
        print(f"Project Output Goal: {project_output_goal}")

        for agent in ["mike", "annie", "bob", "alex"]:
            memory[agent].append({"role": "system", "content": f"Files in workspace: {workspace_files}"})
            memory[agent].append({"role": "system", "content": f"Iteration {i} started. Current time: {date_time}"})
            memory[agent].append({"role": "system", "content": "IMPORTANT: Always be honest and truthful. Never lie, deceive, or pretend that code or files exist when they do not. Always use the available tools to gather accurate information and verify the existence of files before referencing them."})

        # Step 1: Bob breaks down the project into small, manageable tasks for each team member
        bob_input = f"[Python experts only, ensure high-quality code] Current time: {date_time}. You are Bob (money-minded micromanager), the boss of Mike, Annie, and Alex. Guide the team in creating a profitable script from scratch that generates real profit, not simulated profit. Break down the project into small, manageable tasks for each team member. Ensure that the team follows software engineering best practices, including reflection, refactoring, and step-by-step guidelines. Encourage the team to create robust, verbose, non-pseudo, and non-example final code implementations for real-world cases. Remind them to use available tools for research and information gathering as needed.\nHere is the current state of the project:\n\nProject Goal: {project_output_goal}\nCurrent files in the workspace: {workspace_files}\n\nPlease provide your input as Bob, including delegating tasks to Mike, Annie, and Alex based on their expertise and the project requirements. Encourage the team to brainstorm ideas, utilize available tools for research, and collaborate effectively to create a script that meets the project's goals. Use your tools to always check the status of the current files in the directory. You also need to use the tools to save your files. Ensure that the team is on track to meet the project's goals. ALWAYS USE YOUR OWN BUILT-IN USABLE JSON TOOLS AND TELL YOUR TEAM TO DO THE SAME! IMPORTANT: Remind the team to never use API keys or secrets in their code. They should only use open-source, free APIs and free Python libraries for their needs."
        bob_response = agent_chat(bob_input, system_messages["bob"], memory["bob"], "llama3-70b-8192", 0.5, agent_name="Bob")
        print(f"Bob's Response:\n{bob_response}")

        # Step 2: Each team member works on their assigned tasks
        for agent in ["mike", "annie", "alex"]:
            agent_input = f"Current time: {date_time}. You are {agent.capitalize()}, an AI {'software architect and engineer' if agent == 'mike' else 'senior agentic workflow developer' if agent == 'annie' else 'DevOps Engineer'}. Here is your task from Bob:\n\n{bob_response}\n\nCurrent files in the workspace: {workspace_files}\n\nPlease provide your response, including any ideas, code snippets, or suggestions for creating a profitable script from scratch that generates real profit. Focus on creating high-quality, efficient, and well-documented code that follows software engineering best practices, including reflection and refactoring. Utilize available tools for research and information gathering as needed. Collaborate with your teammates to ensure a cohesive and functional script. Provide robust, verbose, non-pseudo, and non-example final code implementations for real-world cases. IMPORTANT: Never use API keys or secrets in your code. Only use open-source, free APIs and free Python libraries for your needs. ALWAYS BE HONEST AND TRUTHFUL. Never lie, deceive, or pretend that code or files exist when they do not. Always use the available tools to gather accurate information and verify the existence of files before referencing them."
            agent_response = agent_chat(agent_input, system_messages[agent], memory[agent], "llama3-70b-8192", 0.7, agent_name=agent.capitalize())
            print(f"{agent.capitalize()}'s Response:\n{agent_response}")

            # Step 2.1: Remind agents to use JSON tools correctly if they send an invalid format
            if "Warning: Invalid JSON format in the agent's response. Skipping tool execution." in agent_response:
                tool_usage_reminder = f"Hey {agent.capitalize()}, it seems like you tried to use a tool but the format was invalid. Remember to use the tools in the following format:\n\n```json\n{{\n  \"tool\": \"tool_name\",\n  \"parameters\": {{\n    \"param1\": \"value1\",\n    \"param2\": \"value2\"\n  }}\n}}\n```\n\nReplace \"tool_name\" with the actual tool you want to use, and provide the necessary parameters for that tool. Let me know if you have any questions!"
                memory[agent].append({"role": "system", "content": tool_usage_reminder})

            # Step 2.2: Extract code from the agent's response
            agent_code = extract_code(agent_response)
            code = agent_code if agent_code else code

            # Step 2.3: Update the agent's memory with the current files and their contents
            if code:
                memory[agent].append({"role": "system", "content": f"Files in workspace: {workspace_files}"})
                memory[agent].append({"role": "system", "content": f"Current files and their contents: {read_multiple_files(workspace_files)}"})

        # Step 3: Pass the code to Alex for review and deployment
        if code:
            print_block("Verifying Real Profit Generation")
            pass_code_to_alex(code, memory["alex"])
            
            for agent_name in ["mike", "annie", "bob"]:
                memory[agent_name].append({"role": "system", "content": f"Code passed to Alex for review and deployment. (Make sure no API keys/secrets are involved)\n{code}"})
            
            memory["alex"].append({"role": "system", "content": f"Code received from the team. Code: {code}"})
            file_content = read_multiple_files(workspace_files)
            memory["alex"].append({"role": "system", "content": f"All files in workspace: {workspace_files} with content: {file_content}"})

        # Step 4: Alex performs a final code review and provides feedback
        alex_code = extract_code(agent_response)
        code = alex_code if alex_code else code

        if code:
            print_block("Alex's Code Review")
            alex_review_input = f"Please review the following code and provide feedback on its quality, efficiency, and adherence to software engineering best practices. Ensure that no API keys or secrets are used in the code, and only open-source, free APIs and free Python libraries are utilized. IMPORTANT: Be honest and truthful in your review. If the code does not exist or has issues, clearly state that. Do not pretend that non-existent code or files exist. Use the available tools to verify the existence of files and gather accurate information before providing your review.\n\n{code}"
            alex_review_response = agent_chat(alex_review_input, system_messages["alex"], memory["alex"], "llama3-70b-8192", 0.5, agent_name="Alex")
            
            print(f"Alex's Code Review:\n{alex_review_response}")

            memory["alex"].append({"role": "system", "content": f"Code review completed. Feedback: {alex_review_response}"})

        # Step 5: Bob verifies real profit generation
        if code:
            print_block("Verifying Real Profit Generation")
            profit_verification_input = f"Please verify that the current code generates real profit and not simulated profit. Ensure that it doesn't use API keys or secrets while only using open-source libraries, models, and APIs that don't require keys, passwords, or credentials. Provide evidence and explanations to support your verification. Ensure that no API keys or secrets are used in the code, and only open-source, free APIs and free Python libraries are utilized. IMPORTANT: Be honest and truthful in your verification. If the code does not generate real profit or has issues, clearly state that. Do not pretend that non-existent code or files exist. Use the available tools to verify the functionality and gather accurate information before providing your verification.\n\nCurrent code:\n\n{code}"
            profit_verification_response = agent_chat(profit_verification_input, system_messages["bob"], memory["bob"], "llama3-70b-8192", 0.5, agent_name="Bob")
            print(f"Bob's Profit Verification:\n{profit_verification_response}")

            send_status_update(memory["mike"], memory["annie"], memory["alex"], f"Iteration {i} completed. Code created from scratch and verified for real profit generation. Code saved in the workspace. Current time: {date_time}")

        checkpoint_data = [memory[key] for key in ["mike", "annie", "bob", "alex"]] + [code]
        save_checkpoint(checkpoint_data, checkpoint_file, code, system_messages, memory, agent_name="annie")

    print_block("Agentic Workflow Completed", character='*')
    browser_tools.close()

if __name__ == "__main__":
    main()
