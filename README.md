
![image](https://github.com/Drlordbasil/GroqAgenticWorkflow/assets/126736516/b010504d-43b8-4faa-91ba-ec0a5ab83091)

# Agentic Workflow

Agentic Workflow is a Python program that simulates a collaborative development process among multiple AI agents. The agents, named Mike, Annie, Alex, and their boss Bob, work together to iteratively improve and refine a codebase. The program showcases how AI agents can assist in software development tasks such as code review, testing, optimization, and deployment.

## Features

- Multi-agent collaboration: The program involves four AI agents (Mike, Annie, Alex, and Bob) working together to improve a codebase.
- Iterative development process: The agents engage in multiple iterations of code review, testing, and refinement.
- Code execution and testing: The program executes the current code and runs tests to identify any errors or issues.
- Code quality checks: It performs code quality checks using tools like pylint to ensure adherence to coding standards.
- Dependency management: The program updates dependencies based on the requirements specified in the `requirements.txt` file.
- Code refactoring and optimization: It applies code refactoring techniques using tools like black and optimizes the code using compilation.
- Deployment: The program simulates the deployment process by executing the final version of the code.
- ![image](https://github.com/Drlordbasil/GroqAgenticWorkflow/assets/126736516/75d3ee9f-ee5f-40ba-9db7-9509ee8056c3)

![image](https://github.com/Drlordbasil/GroqAgenticWorkflow/assets/126736516/e8980c6a-0c07-4f34-a4a8-9a09f9961c40)

## Prerequisites

Before running the Agentic Workflow program, ensure that you have the following prerequisites installed:

- Python 3.x
- pip (Python package installer)

## Installation

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
   pip install -r requirements.txt
   ```

4. Set up the environment variables:

   - Create a `.env` file in the project directory.
   - Add the following variables to the `.env` file:

     ```
     GROQ_API_KEY=your_groq_api_key
     OPENAI_API_KEY=your_openai_api_key
     ```

   - Replace `your_groq_api_key` and `your_openai_api_key` with your actual API keys.

## Usage

To run the Agentic Workflow program, execute the following command:

```
python agentic.py
```

The program will start the collaborative development process among the AI agents. It will iterate through multiple rounds of code review, testing, and refinement. The output will be displayed in the console, showing the interactions between the agents, code execution results, test results, and other relevant information.

## Use Case Examples

1. **Collaborative Code Development**: Agentic Workflow can be used to simulate a collaborative development process where multiple AI agents work together to improve a codebase. The agents can provide suggestions, review code, and iterate on the code to achieve a better result.

2. **Automated Testing and Quality Assurance**: The program demonstrates how AI agents can assist in automated testing and quality assurance tasks. It runs tests, performs code quality checks, and ensures that the code meets the required standards.

3. **Continuous Integration and Deployment**: Agentic Workflow showcases how AI agents can be involved in the continuous integration and deployment process. It simulates the steps of updating dependencies, refactoring code, optimizing code, and deploying the final version.

## Contributing

Contributions to Agentic Workflow are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).
