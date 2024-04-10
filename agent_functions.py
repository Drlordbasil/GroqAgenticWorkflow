from dotenv import load_dotenv
from openai import OpenAI
import datetime
import os
import logging
import re
import pickle
import zlib
from time import sleep

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
            memory.append({"role": "user", "content": "you can use Research topic: (research topic here) to provide a research topic for the agent to work on. Tasks for agent_name: to provide tasks for the agent to work on. You can also provide code for the agent to work on."})
            memory.append({"role": "user", "content": user_input})
            sleep(10)
            return response_content

        except Exception as e:
            #logging.error(f"Error in agent_chat: {format_error_message(e)}")
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
        #logging.error(f"Error extracting code: {format_error_message(e)}")
        return None
def generate_file_name(code):
    try:

        messages = [
            {"role": "system", "content": f" you only respond with an appropriate file name for the code you are given. Dont provide any other information."},
            
            {"role": "user", "content": f" Please provide a suitable file name for the following code:\n\n{code} \n\nFile Name:"}
        ]        
        
        file_name = client["groq_client"].chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=messages,
                max_tokens=20,
                temperature=0,
            )

        
        file_name = file_name.choices[0].message.content
        sleep(10)
        return file_name
    except Exception as e:
        #logging.error(f" {format_error_message(e)}")
        return None
def save_checkpoint(checkpoint_data, checkpoint_file, code):
    try:
        compressed_data = compress_data(checkpoint_data)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(compressed_data, f)

        if code:
            file_name = generate_file_name(code)
            code_file_path = os.path.join("workspace", f"{file_name}.py")
            with open(code_file_path, 'w') as code_file:
                code_file.write(code)
    except Exception as e:
        #logging.error(f"Error saving checkpoint: {format_error_message(e)}")
        pass

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
        #logging.error(f"Error loading checkpoint: {format_error_message(e)}")
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
        #logging.error(f"Error generating summary for {agent_name}: {format_error_message(e)}")
        return None
def extract_task(response, agent_name):
    task_pattern = re.compile(fr'Tasks for {agent_name}:\n1\. (.*?)\n2\.')
    match = task_pattern.search(response)
    return match.group(1) if match else ""

def extract_research_topic(response):
    research_pattern = re.compile(r'Research topic: (.*)')
    match = research_pattern.search(response)
    return match.group(1) if match else ""