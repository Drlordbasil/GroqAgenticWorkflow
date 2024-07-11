import json
import datetime
import os
import re
import pickle
import zlib
from time import sleep
import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from code_execution_manager import CodeExecutionManager
import spacy
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from browser_tools import WebResearchTool
from autogen_coding import AutogenCoding
from task_manager import TaskManager
from memory_ollama import MemoryManager

class AgentFunctions:
    def __init__(self):
        self.code_execution_manager = CodeExecutionManager()
        self.web_research_tool = WebResearchTool()
        self.autogen_coding = AutogenCoding()
        self.task_manager = TaskManager()
        self.memory_manager = MemoryManager()
        self.nlp = spacy.load("en_core_web_sm")
        self.logger = self.setup_logger()
        self.tools = self.load_tools()

    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "save_file",
                    "description": "Save the provided content to a file with the specified file path",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "The content to save to the file"},
                            "file_path": {"type": "string", "description": "The file path to save the content to"}
                        },
                        "required": ["content", "file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the content of the provided file path",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "The file path to read the content from"}
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List the files in the workspace directory",
                    "parameters": {},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information on the provided query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The query to search the web for"}
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "coding",
                    "description": "Start a coding session with the provided task description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string", "description": "The task description for the coding session"}
                        },
                        "required": ["task"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_tasks",
                    "description": "Extract tasks from the given text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The text to extract tasks from"}
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_task_status",
                    "description": "Update the status of a task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "integer", "description": "The ID of the task to update"},
                            "status": {"type": "string", "description": "The new status of the task"}
                        },
                        "required": ["task_id", "status"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_task_summary",
                    "description": "Get a summary of all tasks",
                    "parameters": {},
                },
            },
        ]

    def get_current_date_and_time(self) -> str:
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

    def agent_chat(self, user_input: str, system_message: str, memory: List[Dict[str, str]], 
                   model: str, temperature: float, max_retries: int = 5, 
                   retry_delay: int = 60, agent_name: Optional[str] = None) -> str:
        messages = [
            SystemMessage(content=system_message),
            *[AIMessage(content=msg["content"]) if msg["role"] == "assistant" else HumanMessage(content=msg["content"]) for msg in memory[-3:]],
            HumanMessage(content=user_input)
        ]

        chat = ChatGroq(temperature=temperature, model_name=model)
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | chat

        for retry_count in range(max_retries):
            try:
                self.logger.info(f"Iteration {retry_count + 1} - Engaging {agent_name if agent_name else 'AI Agent'}")

                response_message = chain.invoke({"text": user_input})

                if hasattr(response_message, "content"):
                    self.logger.info(f"{agent_name if agent_name else 'AI Agent'}'s Response:\n{response_message.content}")

                    tool_calls = response_message.tool_calls

                    if tool_calls:
                        messages.append(AIMessage(content=response_message.content))
                        messages.append(AIMessage(content="Tools are available for use. You can use them to perform various tasks. Please wait while I execute the tools."))
                        sleep(10)

                        with ThreadPoolExecutor(max_workers=5) as executor:
                            future_to_tool = {executor.submit(self.execute_tool_call, tool_call): tool_call for tool_call in tool_calls}
                            for future in as_completed(future_to_tool):
                                tool_result = future.result()
                                if tool_result:
                                    messages.append(tool_result)

                        prompt = ChatPromptTemplate.from_messages(messages)
                        chain = prompt | chat
                        response_content = chain.invoke({"text": user_input}).content
                        sleep(10)
                        self.logger.info(f"{agent_name if agent_name else 'AI Agent'}'s Updated Response:\n{response_content}")

                    else:
                        response_content = response_message.content

                    memory.append({"role": "assistant", "content": f"Available tools: {self.tools}"})
                    memory.append({"role": "assistant", "content": response_content})
                    memory.append({"role": "user", "content": user_input})

                    # Prune and summarize memory if it gets too long
                    if len(memory) > 100:
                        summarized_memory = self.summarize_memory(memory)
                        memory.clear()
                        memory.extend(summarized_memory)

                    sleep(20)
                    return response_content

                else:
                    raise ValueError("Response message does not have content attribute.")

            except Exception as e:
                self.logger.error(f"Error encountered: {str(e)}")
                if retry_count < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds... (Attempt {retry_count + 1}/{max_retries})")
                    sleep(retry_delay)
                else:
                    self.logger.error(f"Max retries exceeded. Raising the exception.")
                    raise e

    def execute_tool_call(self, tool_call: Any) -> Optional[Dict[str, Any]]:
        if hasattr(tool_call, "function") and hasattr(tool_call.function, "name") and hasattr(tool_call.function, "arguments"):
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            self.logger.info(f"Executing tool: {function_name}")
            self.logger.info(f"Tool arguments: {function_args}")

            available_functions = {
                "web_search": self.web_research_tool.web_research,
                "save_file": self.code_execution_manager.save_file,
                "read_file": self.code_execution_manager.read_file,
                "list_files": self.code_execution_manager.list_files_in_workspace,
                "coding": self.autogen_coding.start_chat,
                "extract_tasks": self.task_manager.extract_tasks,
                "update_task_status": self.task_manager.update_task_status,
                "get_task_summary": self.task_manager.generate_task_summary,
            }

            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)
                self.logger.info(f"Tool response: {function_response}")

                return {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                }
            else:
                self.logger.warning(f"Unknown tool: {function_name}")
        else:
            self.logger.warning("Invalid tool call format. Skipping tool execution.")
        return None

    def extract_code(self, text: str) -> List[Dict[str, str]]:
        code_blocks = []
        code_block_pattern = re.compile(r'```(\w+)?\n(.*?)```', re.DOTALL)
        matches = code_block_pattern.findall(text)
        for language, code in matches:
            code_blocks.append({
                "language": language if language else "unknown",
                "code": code.strip()
            })
        return code_blocks

    def save_checkpoint(self, checkpoint_data: List[Any], checkpoint_file: str, code: str, 
                        system_messages: Dict[str, str], memory: Dict[str, List[Dict[str, str]]], 
                        agent_name: str = "annie"):
        compressed_data = self.compress_data(checkpoint_data)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(compressed_data, f)

        if code:
            file_name = self.get_file_name_for_code(code, system_messages[agent_name], memory[agent_name], agent_name)
            code_file_path = os.path.join("workspace", file_name)
            with open(code_file_path, 'w') as code_file:
                code_file.write(code)

    def get_file_name_for_code(self, code: str, system_message: str, memory: List[Dict[str, str]], agent_name: str) -> str:
        file_name_response = self.agent_chat(
            f"Please provide a relevant file name for the following code snippet:\n\n{code} \n\n only respond with a singular file name valid for your file. RESPONSE FORMAT ALWAYS(change the filename depending): main.py",
            system_message, memory, "mixtral-8x7b-32768", 0.7, agent_name=agent_name.capitalize()
        )
        file_name_pattern = r'(\w+\.(?:py|txt|json|csv|md))'
        file_name_match = re.search(file_name_pattern, file_name_response, re.IGNORECASE)

        if file_name_match:
            file_name = file_name_match.group(1)
            file_name = file_name.replace("/", "_")
        else:
            file_name = "generated_code.py"

        file_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', file_name)
        file_name = re.sub(r'^\.+|\.+$', '', file_name)
        file_name = re.sub(r'_+', '_', file_name)

        return file_name

    def load_checkpoint(self, checkpoint_file: str) -> Tuple[Optional[List[Any]], str]:
        try:
            with open(checkpoint_file, 'rb') as f:
                compressed_data = pickle.load(f)
                checkpoint_data = self.decompress_data(compressed_data)
                code = checkpoint_data[-1] if checkpoint_data else ""
                return checkpoint_data, code
        except FileNotFoundError:
            self.logger.warning(f"Checkpoint file not found: {checkpoint_file}")
            return None, ""

    def compress_data(self, data: Any) -> bytes:
        return zlib.compress(pickle.dumps(data))

    def decompress_data(self, compressed_data: bytes) -> Any:
        return pickle.loads(zlib.decompress(compressed_data))

    def print_block(self, text: str, width: int = 80, character: str = '='):
        lines = text.split('\n')
        max_line_length = max(len(line) for line in lines)
        padding = (width - max_line_length) // 2

        print(character * width)
        for line in lines:
            print(character + ' ' * padding + line.center(max_line_length) + ' ' * padding + character)
        print(character * width)

    def summarize_memory(self, memory: List[Dict[str, str]]) -> List[Dict[str, str]]:
        summarized_memory = []
        chunk_size = 2000
        for i in range(0, len(memory), chunk_size):
            chunk = memory[i:i+chunk_size]
            chunk_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chunk])
            summary = self.memory_manager.generate_response(
                "Summarize the following conversation chunk, preserving key information:",
                chunk_text
            )
            summarized_memory.append({"role": "system", "content": f"Memory summary: {summary}"})
        return summarized_memory
    def generate_progress_report(self, tasks: List[Dict[str, Any]], code: str) -> str:
        report = "Project Progress Report\n"
        report += "=" * 25 + "\n\n"

        # Task summary
        report += "Task Summary:\n"
        report += "-" * 15 + "\n"
        task_status = {"pending": 0, "in progress": 0, "completed": 0}
        for task in tasks:
            task_status[task.get("status", "pending")] += 1
        for status, count in task_status.items():
            report += f"{status.capitalize()}: {count}\n"
        report += "\n"

        # Recent tasks
        report += "Recent Tasks:\n"
        report += "-" * 13 + "\n"
        for task in tasks[-5:]:
            report += f"- {task.get('task', 'Unnamed task')} ({task.get('status', 'pending')})\n"
        report += "\n"

        # Code summary
        report += "Code Summary:\n"
        report += "-" * 13 + "\n"
        code_lines = code.split("\n")
        report += f"Total lines of code: {len(code_lines)}\n"
        report += f"Functions defined: {len([line for line in code_lines if line.strip().startswith('def ')])}\n"
        report += f"Classes defined: {len([line for line in code_lines if line.strip().startswith('class ')])}\n"
        report += "\n"

        # Recent changes
        report += "Recent Changes:\n"
        report += "-" * 15 + "\n"
        recent_changes = self.get_recent_changes(code)
        for change in recent_changes[-5:]:
            report += f"- {change}\n"

        return report

    def get_recent_changes(self, code: str) -> List[str]:
        # This is a placeholder implementation. In a real-world scenario,
        # you would integrate with a version control system like Git.
        return [
            "Added new function: process_data()",
            "Updated error handling in main loop",
            "Refactored database connection logic",
            "Implemented new feature: user authentication",
            "Fixed bug in task scheduling algorithm"
        ]

    def analyze_code_quality(self, code: str) -> Dict[str, Any]:
        # This method would typically use tools like pylint or flake8
        # For this example, we'll use a simple analysis
        lines = code.split("\n")
        function_count = sum(1 for line in lines if line.strip().startswith("def "))
        class_count = sum(1 for line in lines if line.strip().startswith("class "))
        comment_count = sum(1 for line in lines if line.strip().startswith("#"))
        
        return {
            "total_lines": len(lines),
            "function_count": function_count,
            "class_count": class_count,
            "comment_count": comment_count,
            "comment_ratio": comment_count / len(lines) if len(lines) > 0 else 0
        }

    def execute_code(self, code: str) -> Dict[str, Any]:
        result = self.code_execution_manager.test_code(code)
        if result["status"] == "success":
            return {"success": True, "output": result["output"]}
        else:
            return {"success": False, "error": result["error_message"]}

    def optimize_code(self, code: str) -> Dict[str, Any]:
        optimization_result = self.code_execution_manager.optimize_code(code)
        if optimization_result["status"] == "success":
            return {"success": True, "optimized_code": optimization_result["suggestions"]}
        else:
            return {"success": False, "error": optimization_result["error_message"]}

    def generate_documentation(self, code: str) -> str:
        doc_result = self.code_execution_manager.generate_documentation(code)
        if doc_result["status"] == "success":
            return doc_result["documentation"]
        else:
            return f"Error generating documentation: {doc_result['error_message']}"

    def commit_code_changes(self, code: str, commit_message: str) -> Dict[str, Any]:
        commit_result = self.code_execution_manager.commit_changes(code)
        if commit_result["status"] == "success":
            return {"success": True, "message": commit_result["message"]}
        else:
            return {"success": False, "error": commit_result["error_message"]}

    def get_task_dependencies(self, tasks: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        dependencies = {}
        for i, task in enumerate(tasks):
            task_text = task.get("task", "").lower()
            dependencies[i] = []
            for j, other_task in enumerate(tasks):
                if i != j and other_task.get("task", "").lower() in task_text:
                    dependencies[i].append(j)
        return dependencies

    def prioritize_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dependencies = self.get_task_dependencies(tasks)
        prioritized_tasks = []
        remaining_tasks = set(range(len(tasks)))

        while remaining_tasks:
            next_task = min(remaining_tasks, key=lambda x: len(dependencies[x]))
            prioritized_tasks.append(tasks[next_task])
            remaining_tasks.remove(next_task)
            for deps in dependencies.values():
                if next_task in deps:
                    deps.remove(next_task)

        return prioritized_tasks

    def generate_project_timeline(self, tasks: List[Dict[str, Any]]) -> str:
        prioritized_tasks = self.prioritize_tasks(tasks)
        timeline = "Project Timeline\n"
        timeline += "=" * 17 + "\n\n"

        start_date = datetime.datetime.now()
        current_date = start_date

        for task in prioritized_tasks:
            task_duration = task.get("estimated_duration", 1)  # in days
            timeline += f"{current_date.strftime('%Y-%m-%d')}: {task.get('task', 'Unnamed task')}\n"
            current_date += datetime.timedelta(days=task_duration)

        timeline += f"\nEstimated completion date: {current_date.strftime('%Y-%m-%d')}"
        return timeline

# Example usage
if __name__ == "__main__":
    agent_functions = AgentFunctions()
    current_time = agent_functions.get_current_date_and_time()
    agent_functions.print_block(f"Current Time: {current_time}")
    
    # Example: Generate a progress report
    tasks = [
        {"task": "Design database schema", "status": "completed"},
        {"task": "Implement user authentication", "status": "in progress"},
        {"task": "Create API endpoints", "status": "pending"},
        {"task": "Write unit tests", "status": "pending"},
        {"task": "Deploy to staging environment", "status": "pending"}
    ]
    sample_code = """
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
    """
    progress_report = agent_functions.generate_progress_report(tasks, sample_code)
    print(progress_report)

    # Example: Analyze code quality
    code_quality = agent_functions.analyze_code_quality(sample_code)
    print("Code Quality Analysis:", code_quality)

    # Example: Generate project timeline
    timeline = agent_functions.generate_project_timeline(tasks)
    print(timeline)