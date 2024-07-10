import autogen
from code_execution_manager import CodeExecutionManager

class AutogenCoding:
    def __init__(self, config_list_path="OAI_CONFIG_LIST.json"):
        self.code_execution_manager = CodeExecutionManager()
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
        )

        self.coder = autogen.AssistantAgent(
            name="Coder",
            llm_config=self.llm_config,
            system_message="""You are an expert Python programmer. Write complete, production-ready code. 
            If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line.""",
        )

        self.researcher = autogen.AssistantAgent(
            name="Researcher",
            llm_config=self.llm_config,
            system_message="You are an expert researcher. Use the web_research tool to find information on given topics.",
        )

        self.planner = autogen.AssistantAgent(
            name="Planner",
            llm_config=self.llm_config,
            system_message="You are a project planner. Break down tasks and create workflows using the task_manager.",
        )

        self.group_chat = autogen.GroupChat(
            agents=[self.user_proxy, self.coder, self.researcher, self.planner],
            messages=[],
            max_round=10
        )
        self.manager = autogen.GroupChatManager(groupchat=self.group_chat, llm_config=self.llm_config)

    def start_chat(self, message):
        return self.user_proxy.initiate_chat(self.manager, message=message)

    def get_coding_result(self, task):
        code_message = f"Implement the plan for: {task}"
        return self.user_proxy.initiate_chat(self.coder, message=code_message)

    def get_research_result(self, query):
        research_message = f"Research the following topic: {query}"
        return self.user_proxy.initiate_chat(self.researcher, message=research_message)

    def get_plan(self, task):
        plan_message = f"Create a plan to accomplish: {task}"
        return self.user_proxy.initiate_chat(self.planner, message=plan_message)

    def execute_code(self, code):
        return self.code_execution_manager.test_code(code)

    def save_code(self, filename, code):
        return self.code_execution_manager.save_file(filename, code)

if __name__ == "__main__":
    autogen_coding = AutogenCoding()
    result = autogen_coding.start_chat("Create a simple Python function to calculate the fibonacci sequence.")
    print(result)