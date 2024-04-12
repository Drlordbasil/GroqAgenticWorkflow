import json
import datetime
import os
import re
import pickle
import zlib
from time import sleep
from browser_tools import BrowserTools
from code_execution_manager import CodeExecutionManager
import spacy
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage

def get_current_date_and_time():
    """
    Get the current date and time.

    Returns:
        str: The current date and time in the format 'YYYY-MM-DD HH:MM:SS.ffffff'.
    """
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d %H:%M:%S.%f')

def agent_chat(user_input, system_message, memory, model, temperature, max_retries=5, retry_delay=60, agent_name=None):
    """
    Engage in a conversation with an AI agent using the provided tools and memory.

    Args:
        user_input (str): The user's input message.
        system_message (str): The system message to guide the agent's behavior.
        memory (list): The agent's memory, containing relevant context and information.
        model (str): The name of the language model to use for generating responses.
        temperature (float): The temperature value for controlling the randomness of the generated responses.
        max_retries (int, optional): The maximum number of retries in case of errors. Defaults to 5.
        retry_delay (int, optional): The delay in seconds between retries. Defaults to 60.
        agent_name (str, optional): The name of the agent. Defaults to None.

    Returns:
        str: The agent's response.
    """
    browser_tools = BrowserTools()
    code_execution_manager = CodeExecutionManager()
    nlp = spacy.load("en_core_web_sm")

    messages = [
        SystemMessage(content=system_message),
        *[AIMessage(content=msg["content"]) if msg["role"] == "assistant" else HumanMessage(content=msg["content"]) for msg in memory[-3:]],
        HumanMessage(content=user_input)
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_google",
                "description": "Search Google for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scrape_page",
                "description": "Scrape a web page for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the web page to scrape",
                        }
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "test_code",
                "description": "Test the provided code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to test",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_date_and_time",
                "description": "Get the current date and time",
                "parameters": {},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "save_file",
                "description": "Save the provided content to a file",
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
                "description": "Read the content of the provided file",
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
                "description": "List the files in the workspace",
                "parameters": {},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "format_code",
                "description": "Format the provided code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to format",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_documentation",
                "description": "Generate documentation for the provided code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to generate documentation for",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "commit_changes",
                "description": "Commit changes to the repository",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code changes to commit",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
    ]

    chat = ChatGroq(temperature=temperature, model_name=model)
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | chat

    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"\n{'=' * 80}\n🚀 Iteration {retry_count + 1} - Engaging {agent_name if agent_name else 'AI Agent'} 🚀\n{'=' * 80}\n")

            chat_completion = chain.invoke({"text": user_input})
            response_message = chat_completion.content
            
            print(f"\n🤖 {agent_name if agent_name else 'AI Agent'}'s Response:\n{response_message}\n")

            tool_calls = []
            try:
                tool_calls = json.loads(response_message).get("tool_calls", [])
            except json.JSONDecodeError:
                print("Warning: Invalid JSON format in the agent's response. Skipping tool execution.")

            if tool_calls:
                available_functions = {
                    "search_google": browser_tools.search_google,
                    "scrape_page": browser_tools.scrape_page,
                    "test_code": code_execution_manager.test_code,
                    "extract_code": extract_code,
                    "get_current_date_and_time": get_current_date_and_time,
                    "save_file": code_execution_manager.save_file,
                    "read_file": code_execution_manager.read_file,
                    "list_files": code_execution_manager.list_files_in_workspace,
                    "format_code": code_execution_manager.format_code,
                    "generate_documentation": code_execution_manager.generate_documentation,
                    "commit_changes": code_execution_manager.commit_changes,
                }

                messages.append(AIMessage(content=response_message))
                messages.append(SystemMessage(content="Tools are available for use. You can use them to perform various tasks. Please wait while I execute the tools."))
                sleep(10)
                
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    function_args = tool_call["function"]["arguments"]
                    
                    print(f"🛠️ Executing tool: {function_name}")
                    print(f"📥 Tool arguments: {function_args}")

                    if function_name in available_functions:
                        function_to_call = available_functions[function_name]
                        function_response = function_to_call(**function_args)
                        print(f"📤 Tool response: {function_response}")

                        messages.append(
                            {
                                "tool_call_id": tool_call["id"],
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            }
                        )
                    else:
                        print(f"Unknown tool: {function_name}")
                
                second_response = chain.invoke({"text": user_input})
                response_content = second_response.content
                sleep(10)
                print(f"\n🤖 {agent_name if agent_name else 'AI Agent'}'s Updated Response:\n{response_content}\n")

            else:
                response_content = response_message

            memory.append({"role": "assistant", "content": f"Available tools: {tools}"})
            memory.append({"role": "assistant", "content": response_content})
            memory.append({"role": "user", "content": user_input})
            
            sleep(20)
            return response_content

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

        # Extract the file name using regular expressions
        file_name_pattern = r'(\w+\.(?:py|txt|json|csv|md))'
        file_name_match = re.search(file_name_pattern, file_name_response, re.IGNORECASE)

        if file_name_match:
            file_name = file_name_match.group(1)
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