import os
import json
from agent_functions import AgentFunctions
from code_execution_manager import CodeExecutionManager
from task_manager import TaskManager
from autogen_coding import AutogenCoding
from browser_tools import WebResearchTool
from memory_ollama import MemoryManager

class AgenticWorkflow:
    def __init__(self):
        self.agent_functions = AgentFunctions()
        self.code_execution_manager = CodeExecutionManager()
        self.task_manager = TaskManager()
        self.autogen_coding = AutogenCoding()
        self.web_research_tool = WebResearchTool()
        try:
            self.memory_manager = MemoryManager()
        except Exception as e:
            print(f"Error initializing MemoryManager: {e}")
            print("Continuing without MemoryManager...")
            self.memory_manager = None
        self.checkpoint_file = "agentic_workflow_checkpoint.pkl"
        self.max_iterations = 500
        self.system_messages = self.load_system_messages()
        self.memory = {key: [] for key in ["mike", "annie", "bob", "alex"]}
        self.code = ""

    def load_system_messages(self):
        system_messages = {}
        for agent in ["mike", "annie", "bob", "alex"]:
            filepath = f"system_messages/{agent}.txt"
            with open(filepath, 'r', encoding='utf-8') as file:
                system_messages[agent] = file.read()
        return system_messages

    def run_workflow(self):
        self.agent_functions.print_block("Agentic Workflow", character='*')
        date_time = self.agent_functions.get_current_date_and_time()
        self.agent_functions.print_block(f"Start Time: {date_time}")

        checkpoint_data, self.code = self.agent_functions.load_checkpoint(self.checkpoint_file)
        if checkpoint_data:
            self.memory = {key: value for key, value in zip(["mike", "annie", "bob", "alex"], checkpoint_data)}

        for i in range(1, self.max_iterations + 1):
            self.agent_functions.print_block(f"Iteration {i}")
            self.run_iteration(i, date_time)

            checkpoint_data = [self.memory[key] for key in ["mike", "annie", "bob", "alex"]] + [self.code]
            self.agent_functions.save_checkpoint(checkpoint_data, self.checkpoint_file, self.code, self.system_messages, self.memory, agent_name="annie")

        self.agent_functions.print_block("Agentic Workflow Completed", character='*')

    def run_iteration(self, iteration, date_time):
        workspace_files = self.code_execution_manager.list_files_in_workspace().get("files", [])
        project_output_goal = f"Create a model algorithm to train a model on actionable Windows 11 actions for becoming the first AI real assistant AI. Current time: {date_time} We want 5 total modularized files: main.py, utils.py, training.py, testing.py, and model.py"

        print(f"Project Output Goal: {project_output_goal}")

        # Update agent memories with current state
        for agent in ["mike", "annie", "bob", "alex"]:
            self.memory[agent].append({"role": "assistant", "content": f"Files in workspace: {workspace_files}"})
            self.memory[agent].append({"role": "assistant", "content": f"Iteration {iteration} started. Current time: {date_time}"})
            self.memory[agent].append({"role": "assistant", "content": "IMPORTANT: Always be honest and truthful. Never lie, deceive, or pretend that code or files exist when they do not. Always use the available tools to gather accurate information and verify the existence of files before referencing them."})

        # Bob's task breakdown
        bob_input = self.generate_bob_input(date_time, project_output_goal, workspace_files)
        bob_response = self.agent_functions.agent_chat(bob_input, self.system_messages["bob"], self.memory["bob"], "llama3-70b-8192", 0.5, agent_name="Bob")
        print(f"Bob's Response:\n{bob_response}")

        # Extract tasks from Bob's response
        tasks = self.task_manager.extract_tasks(bob_response)

        # Assign tasks to team members
        for task in tasks:
            assignee = task.get("assignee", "").lower()
            if assignee in ["mike", "annie", "alex"]:
                self.assign_task_to_agent(assignee, task, date_time, workspace_files)

        # Alex's code review and deployment
        if self.code:
            self.perform_code_review_and_deployment()

        # Bob's profit verification
        if self.code:
            self.verify_profit_generation(date_time)

    def generate_bob_input(self, date_time, project_output_goal, workspace_files):
        return f"""
        [Python experts only, ensure high-quality code]
        Current time: {date_time}
        You are Bob (money-minded micromanager), the boss of Mike, Annie, and Alex. Guide the team in creating a profitable script from scratch that generates real profit, not simulated profit.
        Break down the project into small, manageable tasks for each team member. Ensure that the team follows software engineering best practices, including reflection, refactoring, and step-by-step guidelines.
        Encourage the team to create robust, verbose, non-pseudo, and non-example final code implementations for real-world cases. Remind them to use available tools for research and information gathering as needed.

        Here is the current state of the project:
        Project Goal: {project_output_goal}
        Current files in the workspace: {workspace_files}

        Please provide your input as Bob, including delegating tasks to Mike, Annie, and Alex based on their expertise and the project requirements.
        Encourage the team to brainstorm ideas, utilize available tools for research, and collaborate effectively to create a script that meets the project's goals.
        Use your tools to always check the status of the current files in the directory. You also need to use the tools to save your files.
        Ensure that the team is on track to meet the project's goals.
        ALWAYS USE YOUR OWN BUILT-IN USABLE JSON TOOLS AND TELL YOUR TEAM TO DO THE SAME!
        IMPORTANT: Remind the team to never use API keys or secrets in their code. They should only use open-source, free APIs and free Python libraries for their needs.
        """

    def assign_task_to_agent(self, agent, task, date_time, workspace_files):
        agent_input = f"""
        Current time: {date_time}
        You are {agent.capitalize()}, an AI {'software architect and engineer' if agent == 'mike' else 'senior agentic workflow developer' if agent == 'annie' else 'DevOps Engineer'}.
        Here is your task:
        {task}
        tools you have: {self.agent_functions.tools}
        Current files in the workspace: {workspace_files}

        Please provide your response, including any ideas, code snippets, or suggestions for creating a profitable script from scratch that generates real profit.
        Focus on creating high-quality, efficient, and well-documented code that follows software engineering best practices, including reflection and refactoring.
        Utilize available tools for research and information gathering as needed. Collaborate with your teammates to ensure a cohesive and functional script.
        Provide robust, verbose, non-pseudo, and non-example final code implementations for real-world cases.
        IMPORTANT: Never use API keys or secrets in your code. Only use open-source, free APIs and free Python libraries for your needs.
        ALWAYS BE HONEST AND TRUTHFUL. Never lie, deceive, or pretend that code or files exist when they do not.
        Always use the available tools to gather accurate information and verify the existence of files before referencing them.
        """
        agent_response = self.agent_functions.agent_chat(agent_input, self.system_messages[agent], self.memory[agent], "llama3-70b-8192", 0.7, agent_name=agent.capitalize())
        print(f"{agent.capitalize()}'s Response:\n{agent_response}")

        # Extract code from the agent's response
        agent_code = self.agent_functions.extract_code(agent_response)
        if agent_code:
            self.code = agent_code[0]['code'] if agent_code else ""
            self.memory[agent].append({"role": "assistant", "content": f"Code created: {self.code}"})

        # Update the agent's memory with the current files and their contents
        workspace_files = self.code_execution_manager.list_files_in_workspace().get("files", [])
        self.memory[agent].append({"role": "assistant", "content": f"Files in workspace: {workspace_files}"})
        file_contents = self.read_multiple_files(workspace_files)
        self.memory[agent].append({"role": "assistant", "content": f"Current files and their contents: {file_contents}"})

    def perform_code_review_and_deployment(self):
        self.agent_functions.print_block("Alex's Code Review")
        alex_review_input = f"""
        Please review the following code and provide feedback on its quality, efficiency, and adherence to software engineering best practices.
        Ensure that no API keys or secrets are used in the code, and only open-source, free APIs and free Python libraries are utilized.
        IMPORTANT: Be honest and truthful in your review. If the code does not exist or has issues, clearly state that.
        Do not pretend that non-existent code or files exist. Use the available tools to verify the existence of files and gather accurate information before providing your review.

        {self.code}
        """
        alex_review_response = self.agent_functions.agent_chat(alex_review_input, self.system_messages["alex"], self.memory["alex"], "llama3-70b-8192", 0.5, agent_name="Alex")
        print(f"Alex's Code Review:\n{alex_review_response}")

        self.memory["alex"].append({"role": "assistant", "content": f"Code review completed. Feedback: {alex_review_response}"})

    def verify_profit_generation(self, date_time):
        self.agent_functions.print_block("Verifying Real Profit Generation")
        profit_verification_input = f"""
        Please verify that the current code generates real profit and not simulated profit.
        Ensure that it doesn't use API keys or secrets while only using open-source libraries, models, and APIs that don't require keys, passwords, or credentials.
        Provide evidence and explanations to support your verification.
        Ensure that no API keys or secrets are used in the code, and only open-source, free APIs and free Python libraries are utilized.
        IMPORTANT: Be honest and truthful in your verification. If the code does not generate real profit or has issues, clearly state that.
        Do not pretend that non-existent code or files exist. Use the available tools to verify the functionality and gather accurate information before providing your verification.

        Current code:
        {self.code}
        """
        profit_verification_response = self.agent_functions.agent_chat(profit_verification_input, self.system_messages["bob"], self.memory["bob"], "llama3-70b-8192", 0.5, agent_name="Bob")
        print(f"Bob's Profit Verification:\n{profit_verification_response}")

        for agent in ["mike", "annie", "alex"]:
            self.memory[agent].append({"role": "assistant", "content": f"Iteration completed. Code created from scratch and verified for real profit generation. Code saved in the workspace. Current time: {date_time}"})

    def read_multiple_files(self, files):
        content = ""
        for file in files:
            file_content = self.code_execution_manager.read_file(file)
            if file_content and file_content.get("status") == "success":
                content += f"File: {file}\n{file_content.get('content', '')}\n\n"
        return content

if __name__ == "__main__":
    workflow = AgenticWorkflow()
    workflow.run_workflow()