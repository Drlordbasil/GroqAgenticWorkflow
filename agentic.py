import os
import re
import logging
import sys
import pickle
import traceback
from time import sleep
from dotenv import load_dotenv
from openai import OpenAI
from agent_functions import agent_chat, extract_code, extract_research_topic, extract_task, generate_file_name, generate_summary, get_current_date_and_time, load_checkpoint, print_block, save_checkpoint
from code_execution_manager import CodeExecutionManager, format_error_message, run_tests, monitor_performance, optimize_code, format_code, pass_code_to_alex, send_status_update, generate_documentation
import datetime
import zlib
from voice_tools import VoiceTools
from crypto_wallet import CryptoWallet
from browser_tools import BrowserTools  # Import the BrowserTools class



def main():
    max_iterations = 500
    checkpoint_file = "agentic_workflow_checkpoint.pkl"
    code_execution_manager = CodeExecutionManager()
    date_time = get_current_date_and_time()
    voice_tools = VoiceTools()
    browser_tools = BrowserTools()  # Create an instance of BrowserTools

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

    #mike_wallet = CryptoWallet("mike_wallet")
    #annie_wallet = CryptoWallet("annie_wallet")
    #bob_wallet = CryptoWallet("bob_wallet")
    #alex_wallet = CryptoWallet("alex_wallet")

    print_block("Agentic Workflow", character='*')
    print_block(f"Start Time: {date_time}")

    for i in range(1, max_iterations + 1):
        print_block(f"Iteration {i}")
        files = code_execution_manager.list_files_in_workspace()
        # add file list to all memory
        memory["mike"].append({"role": "system", "content": f"files in workspace: {files}"})
        memory["annie"].append({"role": "system", "content": f"files in workspace: {files}"})
        memory["bob"].append({"role": "system", "content": f"files in workspace: {files}"})
        memory["alex"].append({"role": "system", "content": f"files in workspace: {files}"})
        memory["mike"].append({"role": "system", "content": f"Iteration {i} started. Current time: {date_time}"})
        memory["annie"].append({"role": "system", "content": f"Iteration {i} started. Current time: {date_time}"})
        memory["bob"].append({"role": "system", "content": f"Iteration {i} started. Current time: {date_time}"})
        memory["mike"].append({"role": "system", "content": f"Iteration {i} started. Current time: {date_time}"})
        project_output_goal = f"Brainstorm profitable scripts and run them in the workspace. Make sure to remove all placeholders and add real world logic.say Research Topic: <topic> to get info from online about <topic> Current time: {date_time}"

        #mike_wallet_info = mike_wallet.get_wallet_info()
        #annie_wallet_info = annie_wallet.get_wallet_info()
        #bob_wallet_info = bob_wallet.get_wallet_info()
        #alex_wallet_info = alex_wallet.get_wallet_info()
        
        bob_input = f"[python experts only, dont let them write crap files.]Current time:{date_time}You are Bob(money minded micromanager), the boss of Mike, Annie, and Alex.Make sure no one sends code that reverts your progress as code is directly extracted from all responses.\nHere is the current state of the project:\n\nProject Goal: {project_output_goal}\nCurrent code: {code}\nCurrent error: {current_error if 'current_error' in locals() else 'None'}\n\nPlease provide your input as Bob, including delegating tasks to Mike, Annie, and Alex based on their expertise and the project requirements. Provide context and examples to guide them in providing high-quality responses and code snippets that align with the project's goals and best practices. Encourage them to provide detailed explanations and rationale behind their code modifications and suggestions to facilitate better collaboration and knowledge sharing. If you require additional information or resources, you can use the BrowserTools class to research relevant topics and libraries. current files in workspace: {files}. Only code that is directly extracted from the responses of the agents is allowed. Send robust error free, robust, and free of placeholders code."
        rewrite_input = agent_chat(f"as a AI prompt engineering expert,rewrite this to better push coding goals and robust logic{bob_input} ", system_messages["bob"], memory["bob"], "mixtral-8x7b-32768", 0.3)

        bob_response = agent_chat(rewrite_input, system_messages["bob"], memory["bob"], "mixtral-8x7b-32768", 0.5)
        print(f"Bob's Response:\n{bob_response}")
        #bob_summary = generate_summary(bob_response, "Bob")
        #if bob_summary:
            #voice_tools.text_to_speech(bob_summary, "Bob")

        for agent in ["mike", "annie", "alex"]:
            agent_input = f"[Write robust python logic in full form free of placeholders and make sure it isnt crap code]Current time:{date_time}DO NOT CREATE TEST FILES OR USELESS CODE JUST CREATE REAL WORLD CODE THAT CAN BE USED IN A PROJECT FOR REAL WORLD PROFITS.You are {agent.capitalize()}, an AI {'software architect and engineer' if agent == 'mike' else 'senior agentic workflow developer' if agent == 'annie' else 'DevOps Engineer'}. Here is your task from Bob:\n\nTask: {extract_task(bob_response, agent.capitalize())}\n\n\n\nCurrent code: {code}\nCurrent error: {current_error if 'current_error' in locals() else 'None'}\n\nPlease provide your input as {agent.capitalize()}, including detailed explanations and rationale behind your code modifications and suggestions"
            rewrite_input = agent_chat(f"as a AI prompt engineering expert rewrite this to better follow steps for coding{agent_input}", system_messages[agent], memory[agent], "mixtral-8x7b-32768", 0.7)
            agent_response = agent_chat(rewrite_input, system_messages[agent], memory[agent], "mixtral-8x7b-32768", 0.7)
            print(f"{agent.capitalize()}'s Response:\n{agent_response}")
            #agent_summary = generate_summary(agent_response, agent.capitalize())
            #if agent_summary:
                #voice_tools.text_to_speech(agent_summary, agent.capitalize())
            agent_code = extract_code(agent_response)
            agent_code = agent_chat(f"Refactor this code and improve it removing any and all placeholders:\n\n {agent_code}", system_messages[agent], memory[agent], "mixtral-8x7b-32768", 0.7)
            agent_code = extract_code(agent_code)
            if agent in ["mike", "annie"]:
                pass_code_to_alex(agent_code, memory["alex"])

            elif agent == "alex":
                code = agent_code or code
                current_error = code_execution_manager.test_code(code)[1]
                if current_error:
                    memory["alex"].append({"role": "system", "content": f"Error in code: {current_error} code not saved."})
                else:
                    
                    file_name = generate_file_name(code)
                    code_execution_manager.save_file(file_name, code)
                    memory["alex"].append({"role": "system", "content": f"Code execution successful and error free. Code saved in workspace as {file_name}."})

            # Perform research using BrowserTools if requested by the agent
            research_topic = extract_research_topic(agent_response)
            if research_topic:
                research_results = browser_tools.research_topic(research_topic)
                research_summary = "\n".join([f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content'][:1000]}..." for result in research_results])
                memory[agent].append({"role": "user", "content": f"Research results for topic '{research_topic}':\n{research_summary}"})
        
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
            file_name = generate_file_name(code)
            code_execution_manager.save_file(file_name, code)

            send_status_update(memory["mike"], memory["annie"], memory["alex"], f"Iteration {i} completed. Code updated and tested. Add more logic, remove all placeholders. all testing info: {test_results}. Performance data: {performance_data}. Optimization suggestions: {optimization_suggestions}. Documentation: {documentation} saved under file name: {file_name} in workspace. Code for {file_name} is: \n\n {code}")
        
        #mike_wallet.get_wallet_info()
        #annie_wallet.get_wallet_info()
        #bob_wallet.get_wallet_info()
        #alex_wallet.get_wallet_info()
        
        checkpoint_data = [memory[key] for key in ["mike", "annie", "bob", "alex"]] + [code]
        save_checkpoint(checkpoint_data, checkpoint_file, code)

    print_block("Agentic Workflow Completed", character='*')
    browser_tools.close()  # Close the browser



if __name__ == "__main__":
    main()
