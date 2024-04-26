

# Agentic Workflow README

Welcome to the Agentic Workflow project! This project aims to create an AI-powered solution that generates profitable Python scripts through collaboration between AI agents. The agents work together to break down tasks, write code, review and refactor, and ensure the generated scripts are efficient, well-documented, and capable of producing real profit.
![image](https://github.com/Drlordbasil/GroqAgenticWorkflow/assets/126736516/52cfbcf7-2f86-4acf-a6a5-1fdc397f378c)

## Project Overview

The Agentic Workflow project consists of the following key components:

- `agentic.py`: The main script that orchestrates the collaboration between AI agents.
- `agent_functions.py`: Contains utility functions used by the agents during the workflow.
- `code_execution_manager.py`: Manages code execution, testing, optimization, and documentation generation.
- `browser_tools.py`: Provides tools for web browser interaction and web scraping.
- `crypto_wallet.py`: Implements a cryptocurrency wallet for handling transactions.
- `task_manager.py`: Manages and tracks tasks assigned to the agents.
- `system_messages/`: Contains system messages that guide the behavior of each agent.

## Getting Started

To run the Agentic Workflow project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/Drlordbasil/GroqAgenticWorkflow.git
   ```

2. Navigate to the project directory:
   ```
   cd GroqAgenticWorkflow
   ```

3. Install the required dependencies:
   ```
   pip install langchain langchain_groq spacy requests beautifulsoup4 selenium webdriver_manager pytest pylint black bitcoinlib
   ```

4. Run the `agentic.py` script:
   ```
   python agentic.py
   ```

The script will initiate the collaboration between the AI agents, and you can monitor the progress and generated code in the console output.

## AI Agents

The Agentic Workflow project involves the following AI agents:

- **Bob (Project Manager Extraordinaire)**: Leads the team, breaks down the project into manageable tasks, and assigns them to the other agents. Bob ensures the project stays on track and meets its goals.

- **Mike (AI Software Architect and Engineer)**: Responsible for code analysis, feature development, algorithm design, and code quality assurance. Mike infuses the project with cutting-edge AI capabilities.

- **Annie (Senior Agentic Workflow Developer)**: Focuses on user interface design, workflow optimization, error handling, and cross-platform compatibility. Annie creates intuitive and efficient workflows.

- **Alex (DevOps Engineer Mastermind)**: Handles environment setup, code execution, testing, deployment, and maintenance. Alex ensures the project runs smoothly and efficiently.

## JSON Tools

The agents utilize JSON tools to automate tasks, gather information, and enhance the agentic workflow solution. Some of the key JSON tools include:

- `search_google`: Searches Google for relevant information.
- `scrape_page`: Scrapes a web page for relevant information.
- `test_code`: Tests the provided code.
- `optimize_code`: Optimizes the provided code and offers suggestions.
- `generate_documentation`: Generates documentation for the provided code.
- `execute_browser_command`: Executes a browser command for web interaction.

Agents can invoke these tools using specific JSON formats within their responses.

## Contributing

We welcome contributions to the Agentic Workflow project! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your forked repository.
5. Submit a pull request detailing your changes.

Please ensure that your code adheres to the project's coding standards and includes appropriate documentation.

## License

The Agentic Workflow project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions, suggestions, or feedback, please feel free to contact the project maintainer, Anthony Snider (Drlordbasil), at [drlordbasil@gmail.com](mailto:drlordbasil@gmail.com).

Happy coding!
