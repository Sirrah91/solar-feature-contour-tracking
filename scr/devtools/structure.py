import os
import ast


def get_project_structure(
        folder_path: str
) -> dict:
    """
    Returns a dictionary representing the structure of a Python project.
    Folder hierarchy -> files -> list of function names.
    Nested functions are prefixed with '>'.
    """

    EXCLUDED_DIRS = ["log", "OpenPBS", "graphic_output", ".idea", "python_compiled", "__pycache__", "tests"]

    def extract_functions_from_file(file_path: str) -> list[str]:
        """Return a list of function names in the Python file, marking nested functions."""
        function_names: list[str] = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            def visit(node, depth: int = 0):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.FunctionDef):
                        prefix = ">" * depth
                        function_names.append(f"{prefix}{child.name}")
                        # Recurse into this function, increasing depth
                        visit(child, depth + 1)
                    else:
                        visit(child, depth)

            visit(tree)

        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")

        return function_names

    def build_structure(path: str) -> dict:
        structure = {}
        for entry in os.scandir(path):
            if entry.is_dir() and entry.name not in EXCLUDED_DIRS:
                structure[entry.name] = build_structure(entry.path)
            elif entry.is_file() and entry.name.endswith(".py"):
                structure[entry.name] = extract_functions_from_file(entry.path)
        return structure

    return build_structure(folder_path)


if __name__ == "__main__":
    from pprint import pprint

    pprint(get_project_structure("./"))
