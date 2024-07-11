import json
import datetime
import os
import re
import pickle
import zlib
from time import sleep

from code_execution_manager import CodeExecutionManager
import spacy
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from browser_tools import WebResearchTool
from autogen_coding import AutogenCoding
tools = [


        {
            "type": "function",
            "function": {
                "name": "save_file",
                "description": "Save the provided content to a file with the specified file path",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to save to the file",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "The file path to save the content to",
                        }
                    },
                    "required": ["content", "file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the content of the provided file path and return the content as a string",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The file path to read the content from",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List the files in the workspace directory and return the list of file names as a string array",
                "parameters": {},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information on the provided query and return the search results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search the web for",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "coding",
                "description": "Start a coding session with the provided task description and return the coding result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task description for the coding session",
                        }
                    },
                    "required": ["task"],
                },
            },
        },

    ]
def get_current_date_and_time():
    """
    Get the current date and time.

    Returns:
        str: The current date and time in the format 'YYYY-MM-DD HH:MM:SS.ffffff'.
    """
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d %H:%M:%S.%f')



def agent_chat(user_input, system_message, memory, model, temperature, max_retries=5, retry_delay=60, agent_name=None):
    
    code_execution_manager = CodeExecutionManager()
    web_search = WebResearchTool()
    coding = AutogenCoding()
    messages = [
        SystemMessage(content=system_message),
        *[AIMessage(content=msg["content"]) if msg["role"] == "assistant" else HumanMessage(content=msg["content"]) for msg in memory[-3:]],
        HumanMessage(content=user_input)
    ]




    chat = ChatGroq(temperature=temperature, model_name=model)
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | chat

    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"\n{'=' * 80}\n🚀 Iteration {retry_count + 1} - Engaging {agent_name if agent_name else 'AI Agent'} 🚀\n{'=' * 80}\n")

            response_message = chain.invoke({"text": user_input})

            if hasattr(response_message, "content"):
                print(f"\n🤖 {agent_name if agent_name else 'AI Agent'}'s Response:\n{response_message.content}\n")

                tool_calls = response_message.tool_calls

                if tool_calls:
                    available_functions = {
                        "web_search": web_search.web_research,
                        "save_file": code_execution_manager.save_file,
                        "read_file": code_execution_manager.read_file,
                        "list_files": code_execution_manager.list_files_in_workspace,
                        "coding": coding.start_chat,

                    }

                    messages.append(AIMessage(content=response_message.content))
                    messages.append(AIMessage(content="Tools are available for use. You can use them to perform various tasks. Please wait while I execute the tools."))
                    sleep(10)

                    for tool_call in tool_calls:
                        if hasattr(tool_call, "function") and hasattr(tool_call.function, "name") and hasattr(tool_call.function, "arguments"):
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)

                            print(f"🛠️ Executing tool: {function_name}")
                            print(f"📥 Tool arguments: {function_args}")

                            if function_name in available_functions:
                                function_to_call = available_functions[function_name]
                                function_response = function_to_call(**function_args)
                                print(f"📤 Tool response: {function_response}")

                                messages.append(
                                    {
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": function_response,
                                    }
                                )
                            else:
                                print(f"Unknown tool: {function_name}")
                        else:
                            print("Warning: Invalid tool call format. Skipping tool execution.")

                    prompt = ChatPromptTemplate.from_messages(messages)
                    chain = prompt | chat
                    response_content = chain.invoke({"text": user_input}).content
                    sleep(10)
                    print(f"\n🤖 {agent_name if agent_name else 'AI Agent'}'s Updated Response:\n{response_content}\n")

                else:
                    response_content = response_message.content

                memory.append({"role": "assistant", "content": f"Available tools: {tools}"})
                memory.append({"role": "assistant", "content": response_content})
                memory.append({"role": "user", "content": user_input})

                sleep(20)
                return response_content

            else:
                print("Warning: Response message does not have content attribute. Retrying.")
                raise Exception("Response message does not have content attribute.")

        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"❌ Error encountered: {str(e)}")
                print(f"🔄 Retrying in {retry_delay} seconds... (Attempt {retry_count}/{max_retries})")
                sleep(retry_delay)
            else:
                print(f"❌ Max retries exceeded. Raising the exception.")
                raise e

def extract_code(text):
    """
    Extract code blocks from the provided text.

    Args:
        text (str): The text to extract code blocks from.

    Returns:
        str: The extracted code block, or None if no code block is found.
    """
    code_block_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    code_blocks = code_block_pattern.findall(text)
    return code_blocks[0].strip() if code_blocks else None
def save_checkpoint(checkpoint_data, checkpoint_file, code, system_messages, memory, agent_name="annie"):
    """
    Save the checkpoint data and code to files.

    Args:
        checkpoint_data (list): The checkpoint data to be saved.
        checkpoint_file (str): The path to the checkpoint file.
        code (str): The code to be saved.
        system_messages (dict): The dictionary containing system messages for each agent.
        memory (dict): The dictionary containing the memory for each agent.
        agent_name (str, optional): The name of the agent responsible for naming the code file. Defaults to "annie".

    Returns:
        None
    """
    compressed_data = compress_data(checkpoint_data)
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(compressed_data, f)

    if code:
        # Engage the agent to provide a relevant file name
        file_name_response = agent_chat(f"Please provide a relevant file name for the following code snippet:\n\n{code} \n\n only respond with a singular file name valid for your file. RESPONSE FORMAT ALWAYS(change the filename depending): main.py", system_messages[agent_name], memory[agent_name], "mixtral-8x7b-32768", 0.7, agent_name=agent_name.capitalize())
        # remove /n from the response
        # Extract the file name using regular expressions
        file_name_pattern = r'(\w+\.(?:py|txt|json|csv|md))'
        file_name_match = re.search(file_name_pattern, file_name_response, re.IGNORECASE)

        if file_name_match:
            file_name = file_name_match.group(1)
            file_name = file_name.replace("/", "_")
        else:
            # If no valid file name is found, use a default name
            file_name = "generated_code.py"

        # Sanitize the file name to remove any invalid characters and replace them with underscores
        file_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', file_name)

        # Remove any leading or trailing periods and replace multiple consecutive underscores with a single underscore
        file_name = re.sub(r'^\.+|\.+$', '', file_name)
        file_name = re.sub(r'_+', '_', file_name)

        code_file_path = os.path.join("workspace", file_name)
        with open(code_file_path, 'w') as code_file:
            code_file.write(code)

def load_checkpoint(checkpoint_file):
    """
    Load the checkpoint data and code from files.

    Args:
        checkpoint_file (str): The path to the checkpoint file.

    Returns:
        tuple: A tuple containing the checkpoint data and code.
            - checkpoint_data (list): The loaded checkpoint data.
            - code (str): The loaded code.
    """
    try:
        with open(checkpoint_file, 'rb') as f:
            compressed_data = pickle.load(f)
            checkpoint_data = decompress_data(compressed_data)
            code = checkpoint_data[-1] if checkpoint_data else ""
            return checkpoint_data, code
    except FileNotFoundError:
        return None, ""

def compress_data(data):
    """
    Compress the provided data using zlib compression.

    Args:
        data (object): The data to be compressed.

    Returns:
        bytes: The compressed data.
    """
    return zlib.compress(pickle.dumps(data))

def decompress_data(compressed_data):
    """
    Decompress the provided compressed data using zlib decompression.

    Args:
        compressed_data (bytes): The compressed data to be decompressed.

    Returns:
        object: The decompressed data.
    """
    return pickle.loads(zlib.decompress(compressed_data))

def print_block(text, width=80, character='='):
    """
    Print the provided text in a block format with a specified width and character.

    Args:
        text (str): The text to be printed in the block.
        width (int, optional): The width of the block. Defaults to 80.
        character (str, optional): The character to use for the block border. Defaults to '='.

    Returns:
        None
    """
    lines = text.split('\n')
    max_line_length = max(len(line) for line in lines)
    padding = (width - max_line_length) // 2

    print(character * width)
    for line in lines:
        print(character + ' ' * padding + line.center(max_line_length) + ' ' * padding + character)
    print(character * width)