import os
import ast
import polars as pl
from pathlib import Path

def get_imports(file_path: str) -> list[dict]:
    """Extract imports with modules, names, and aliases using AST parsing."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "imported_name": alias.name,
                    "alias": alias.asname
                })
        elif isinstance(node, ast.ImportFrom):
            if node.module:  # Include all imports
                for alias in node.names:
                    imports.append({
                        "module": node.module,
                        "imported_name": alias.name,
                        "alias": alias.asname
                    })
    return imports

def analyze_projects(parent_directory: str, output_directory: str) -> None:
    """Analyze all projects under parent directory and save imports to separate CSV files."""
    os.makedirs(output_directory, exist_ok=True)
    
    project_folders = [
        folder for folder in os.listdir(parent_directory)
        if os.path.isdir(os.path.join(parent_directory, folder))
    ]
    
    for project in project_folders:
        project_path = os.path.join(parent_directory, project)
        data = []
        
        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    for imp in get_imports(file_path):
                        data.append({
                            "file_path": os.path.relpath(file_path, project_path),
                            **imp
                        })
        
        if data:
            output_csv = os.path.join(output_directory, f"{project}_imports.csv")
            df = pl.DataFrame(data)
            df.write_csv(output_csv)
            print(f"Import analysis for {project} saved to {output_csv}")
        else:
            print(f"No Python files or imports found in project: {project}")

parent_directory = "../dataset"  
output_directory = "import reports"  

analyze_projects(parent_directory, output_directory)