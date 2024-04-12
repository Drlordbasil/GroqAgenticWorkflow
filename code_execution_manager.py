import os
import subprocess
import tempfile
import logging
import cProfile
import pstats
import io
import ast
import astroid
import pylint.lint
import traceback
import pytest
from pylint import config

class CodeExecutionManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.workspace_folder = "workspace"
        os.makedirs(self.workspace_folder, exist_ok=True)

    def save_file(self, filepath, content):
        """
        Save a file with the given content in the workspace folder.

        Args:
            filepath (str): The path of the file relative to the workspace folder.
            content (str): The content to be written to the file.

        Returns:
            dict: A dictionary containing the status and file path.
                - status (str): "success" if the file was saved successfully, "error" otherwise.
                - file_path (str): The full path of the saved file.
        """
        file_path = os.path.join(self.workspace_folder, filepath)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            self.logger.info(f"File '{file_path}' saved successfully.")
            return {"status": "success", "file_path": file_path}
        except Exception as e:
            self.logger.exception(f"Error saving file '{file_path}': {str(e)}")
            return {"status": "error", "error_message": str(e)}

    def read_file(self, filepath):
        """
        Read the content of a file from the workspace folder.

        Args:
            filepath (str): The path of the file relative to the workspace folder.

        Returns:
            dict: A dictionary containing the status, file content, and file path.
                - status (str): "success" if the file was read successfully, "error" otherwise.
                - content (str): The content of the file.
                - file_path (str): The full path of the read file.
        """
        file_path = os.path.join(self.workspace_folder, filepath)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            self.logger.info(f"File '{file_path}' read successfully.")
            return {"status": "success", "content": content, "file_path": file_path}
        except FileNotFoundError:
            self.logger.error(f"File '{file_path}' not found.")
            return {"status": "error", "error_message": f"File '{file_path}' not found."}
        except Exception as e:
            self.logger.exception(f"Error reading file '{file_path}': {str(e)}")
            return {"status": "error", "error_message": str(e)}

    def list_files_in_workspace(self):
        """
        List all the files in the workspace folder.

        Returns:
            dict: A dictionary containing the status and list of files.
                - status (str): "success" if the files were listed successfully, "error" otherwise.
                - files (list): A list of file names in the workspace folder.
        """
        try:
            files = os.listdir(self.workspace_folder)
            self.logger.info("List of files in workspace retrieved successfully.")
            return {"status": "success", "files": files}
        except Exception as e:
            self.logger.exception(f"Error listing files in workspace: {str(e)}")
            return {"status": "error", "error_message": str(e)}

    def test_code(self, code):
        """
        Run tests on the provided code using pytest.

        Args:
            code (str): The code to be tested.

        Returns:
            dict: A dictionary containing the status, test output, and error message (if any).
                - status (str): "success" if the tests passed, "failure" if the tests failed, "error" if an error occurred.
                - output (str): The output of the test execution.
                - error_message (str): The error message if an error occurred during test execution.
        """
        if not code:
            return {"status": "error", "error_message": "No code provided."}

        with tempfile.TemporaryDirectory(dir=self.workspace_folder) as temp_dir:
            script_path = os.path.join(temp_dir, 'temp_script.py')
            with open(script_path, 'w') as f:
                f.write(code)

            try:
                # Use pytest to run tests and capture output
                result = pytest.main([script_path, "--verbose"])
                
                if result == 0:
                    self.logger.info("Tests execution successful.")
                    return {"status": "success", "output": "All tests passed."}
                else:
                    self.logger.error("Tests execution failed.")
                    return {"status": "failure", "output": "Some tests failed."}
            except subprocess.TimeoutExpired:
                self.logger.error("Tests execution timed out after 30 seconds.")
                return {"status": "error", "error_message": "Execution timed out after 30 seconds."}
            except Exception as e:
                self.logger.exception(f"Tests execution error: {str(e)}")
                return {"status": "error", "error_message": f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"}

    def execute_command(self, command):
        """
        Execute a shell command.

        Args:
            command (str): The command to be executed.

        Returns:
            dict: A dictionary containing the status, stdout, and stderr.
                - status (str): "success" if the command executed successfully, "error" otherwise.
                - stdout (str): The standard output of the command execution.
                - stderr (str): The standard error of the command execution.
        """
        try:
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            self.logger.info(f"Command executed: {command}")
            return {"status": "success", "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            self.logger.exception(f"Error executing command: {str(e)}")
            return {"status": "error", "error_message": str(e)}

    def optimize_code(self, code):
        """
        Optimize the provided code using Pylint and provide optimization suggestions.

        Args:
            code (str): The code to be optimized.

        Returns:
            dict: A dictionary containing the status and optimization suggestions.
                - status (str): "success" if the optimization completed successfully, "error" otherwise.
                - suggestions (str): The optimization suggestions provided by Pylint.
        """
        try:
            # Save the code to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
                tmp.write(code.encode('utf-8'))
                tmp_file_path = tmp.name

            # Configure Pylint with custom settings
            pylint_config_path = pylint.config.find_pylintrc()
            pylint_args = [tmp_file_path, "--rcfile", pylint_config_path]

            # Run Pylint analysis
            pylint_output = subprocess.check_output(["pylint"] + pylint_args, universal_newlines=True)

            # Parse Pylint output and provide actionable suggestions
            suggestions = []
            for line in pylint_output.splitlines():
                if line.startswith("C:") or line.startswith("R:") or line.startswith("W:"):
                    msg_id, _, msg = line.partition(":")
                    suggestion = f"Suggestion: {msg.strip()}"
                    if msg_id.startswith("C:"):
                        suggestion += " (Convention Violation)"
                    elif msg_id.startswith("R:"):
                        suggestion += " (Refactoring Opportunity)"
                    elif msg_id.startswith("W:"):
                        suggestion += " (Potential Bug)"
                    suggestions.append(suggestion)

            # Cleanup temporary file
            os.remove(tmp_file_path)

            if suggestions:
                optimization_suggestions = "\n".join(suggestions)
                self.logger.info(f"Optimization suggestions:\n{optimization_suggestions}")
                return {"status": "success", "suggestions": optimization_suggestions}
            else:
                self.logger.info("No optimization suggestions found.")
                return {"status": "success", "suggestions": "No optimization suggestions found."}

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Pylint analysis failed: {e.output}")
            return {"status": "error", "error_message": str(e)}

        except Exception as e:
            self.logger.exception(f"Error during optimization: {str(e)}")
            return {"status": "error", "error_message": str(e)}

    def format_code(self, code):
        """
        Format the provided code using Black code formatter.

        Args:
            code (str): The code to be formatted.

        Returns:
            dict: A dictionary containing the status and formatted code.
                - status (str): "success" if the formatting completed successfully, "error" otherwise.
                - formatted_code (str): The formatted code.
        """
        try:
            # Save the code to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
                tmp.write(code.encode('utf-8'))
                tmp_file_path = tmp.name

            # Run Black code formatter
            subprocess.run(["black", tmp_file_path], check=True)

            # Read the formatted code from the temporary file
            with open(tmp_file_path, "r", encoding="utf-8") as f:
                formatted_code = f.read()

            # Cleanup temporary file
            os.remove(tmp_file_path)

            self.logger.info("Code formatting completed.")
            return {"status": "success", "formatted_code": formatted_code}

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Code formatting failed: {e.output}")
            return {"status": "error", "error_message": str(e)}

        except Exception as e:
            self.logger.exception(f"Error during code formatting: {str(e)}")
            return {"status": "error", "error_message": str(e)}

    def generate_documentation(self, code):
        """
        Generate documentation for the provided code using docstrings.

        Args:
            code (str): The code to generate documentation for.

        Returns:
            dict: A dictionary containing the status and generated documentation.
                - status (str): "success" if the documentation generation completed successfully, "error" otherwise.
                - documentation (str): The generated documentation.
        """
        try:
            module = ast.parse(code)
            docstrings = []

            for node in ast.walk(module):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        docstrings.append(f"{node.name}:\n{docstring}")

            documentation = "\n".join(docstrings)
            self.logger.info(f"Documentation generated:\n{documentation}")
            return {"status": "success", "documentation": documentation}

        except SyntaxError as e:
            self.logger.error(f"SyntaxError: {e}")
            return {"status": "error", "error_message": str(e)}

        except Exception as e:
            self.logger.exception(f"Error during documentation generation: {str(e)}")
            return {"status": "error", "error_message": str(e)}

    def commit_changes(self, code):
        """
        Commit code changes to the version control system.

        Args:
            code (str): The code changes to be committed.

        Returns:
            dict: A dictionary containing the status and commit message.
                - status (str): "success" if the commit completed successfully, "error" otherwise.
                - message (str): The commit message.
        """
        try:
            # Save the code changes to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
                tmp.write(code.encode('utf-8'))
                tmp_file_path = tmp.name

            # Stage the changes
            subprocess.run(["git", "add", tmp_file_path], check=True)

            # Commit the changes with a message
            commit_message = "Automated code commit"
            subprocess.run(["git", "commit", "-m", commit_message], check=True)

            # Push the changes to the remote repository
            subprocess.run(["git", "push"], check=True)

            # Cleanup temporary file
            os.remove(tmp_file_path)

            self.logger.info("Code changes committed successfully.")
            return {"status": "success", "message": commit_message}

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Commit failed: {e.output}")
            return {"status": "error", "error_message": str(e)}

        except Exception as e:
            self.logger.exception(f"Error during commit: {str(e)}")
            return {"status": "error", "error_message": str(e)}