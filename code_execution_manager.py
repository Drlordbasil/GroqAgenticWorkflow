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
class CodeExecutionManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.workspace_folder = "workspace"
        os.makedirs(self.workspace_folder, exist_ok=True)

    def save_file(self, filepath, content):
        filepath = os.path.join(self.workspace_folder, filepath)
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(content)
            self.logger.info(f"File '{filepath}' saved successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error saving file '{filepath}': {str(e)}")
            return False

    def read_file(self, filepath):
        filepath = os.path.join(self.workspace_folder, filepath)
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            self.logger.info(f"File '{filepath}' read successfully.")
            return content
        except FileNotFoundError:
            self.logger.error(f"File '{filepath}' not found.")
            return None
        except Exception as e:
            self.logger.error(f"Error reading file '{filepath}': {str(e)}")
            return None

    def test_code(self, code):
        if not code:
            return None, None

        with tempfile.TemporaryDirectory(dir=self.workspace_folder) as temp_dir:
            script_path = os.path.join(temp_dir, 'temp_script.py')
            with open(script_path, 'w') as f:
                f.write(code)

            try:
                output = subprocess.check_output(['python', '-m', 'unittest', 'discover', temp_dir], universal_newlines=True, stderr=subprocess.STDOUT, timeout=30)
                self.logger.info("Tests execution successful.")
                return output, None
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Tests execution error: {e.output}")
                return None, e.output
            except subprocess.TimeoutExpired:
                self.logger.error("Tests execution timed out after 30 seconds.")
                return None, "Execution timed out after 30 seconds"
            except Exception as e:
                self.logger.error(f"Tests execution error: {str(e)}")
                return None, str(e)

    def execute_command(self, command):
        try:
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            self.logger.info(f"Command executed: {command}")
            return result.stdout, result.stderr
        except Exception as e:
            self.logger.error(f"Error executing command: {str(e)}")
            return None, str(e)

def format_error_message(error):
    return f"Error: {str(error)}\nTraceback: {traceback.format_exc()}"

def run_tests(code):
    code_execution_manager = CodeExecutionManager()
    test_code_output, test_code_error = code_execution_manager.test_code(code)
    if test_code_output:
        print(f"\n[TEST CODE OUTPUT]\n{test_code_output}")
    if test_code_error:
        print(f"\n[TEST CODE ERROR]\n{test_code_error}")

def monitor_performance(code):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir="workspace") as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        subprocess.run(['python', temp_file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing code: {e}")
    finally:
        profiler.disable()
        os.unlink(temp_file_path)

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
    stats.print_stats()

    performance_data = stream.getvalue()
    print(f"\n[PERFORMANCE DATA]\n{performance_data}")

    return performance_data

def optimize_code(code):
    try:
        # Save the code to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
            tmp.write(code.encode('utf-8'))
            tmp_file_path = tmp.name

        # Setup Pylint to use the temporary file
        pylint_output = io.StringIO()

        # Define a custom reporter class based on BaseReporter
        class CustomReporter(pylint.reporters.BaseReporter):
            def _display(self, layout):
                pylint_output.write(str(layout))

        pylint_args = [tmp_file_path]
        pylint_reporter = pylint.lint.Run(pylint_args, reporter=CustomReporter(), do_exit=False)

        # Retrieve optimization suggestions
        optimization_suggestions = pylint_output.getvalue()
        print(f"\n[OPTIMIZATION SUGGESTIONS]\n{optimization_suggestions}")

        # Cleanup temporary file
        os.remove(tmp_file_path)

        return optimization_suggestions
    except SyntaxError as e:
        print(f"SyntaxError: {e}")
        return None
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        return None



def pass_code_to_alex(code, alex_memory):
    alex_memory.append({"role": "system", "content": f"Code from Mike and Annie: {code}"})

def send_status_update(mike_memory, annie_memory, alex_memory, project_status):
    mike_memory.append({"role": "system", "content": f"Project Status Update: {project_status}"})
    annie_memory.append({"role": "system", "content": f"Project Status Update: {project_status}"})
    alex_memory.append({"role": "system", "content": f"Project Status Update: {project_status}"})

def generate_documentation(code):
    try:
        module = ast.parse(code)
        docstrings = []

        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    docstrings.append(f"{node.name}:\n{docstring}")

        documentation = "\n".join(docstrings)
        print(f"\n[GENERATED DOCUMENTATION]\n{documentation}")

        return documentation
    except SyntaxError as e:
        print(f"SyntaxError: {e}")
        return None
def commit_changes(code):
    subprocess.run(["git", "add", "workspace"])
    subprocess.run(["git", "commit", "-m", "Automated code commit"])
