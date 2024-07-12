import os
import autogen

class AutogenCoding:
    def __init__(self, config_list_path="OAI_CONFIG_LIST.json"):
        self.config_list = autogen.config_list_from_json(env_or_file=config_list_path)
        self.llm_config = {
            "cache_seed": 42,
            "temperature": 0,
            "config_list": self.config_list,
            "timeout": 120,
        }
        self._initialize_agents()

    def _initialize_agents(self):
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            system_message="Executor. Execute the code written by the coder and suggest updates if there are errors.",
            human_input_mode="NEVER",
            code_execution_config={
                "last_n_messages": 3,
                "work_dir": "workspace",
                "use_docker": False,
            },
            max_consecutive_auto_reply=5  # Limit the number of consecutive auto-replies
        )

        self.coder = autogen.AssistantAgent(
            name="Coder",
            llm_config=self.llm_config,
            system_message="""You are an expert Python programmer. Write complete, production-ready code. 
            If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line.""",
        )

    def start_chat(self, message):
        return self.user_proxy.initiate_chat(self.coder, message=message, max_turns=10)  # Limit the number of turns

if __name__ == "__main__":
    autogen_coding = AutogenCoding()
    result = autogen_coding.start_chat("Create a simple Python function to calculate the Fibonacci sequence.")
    print(result)
