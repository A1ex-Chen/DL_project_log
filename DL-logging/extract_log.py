import pandas as pd
import os
import re
from pathlib import Path
import ast

def get_full_logging_statement(lines, start_line_idx, start_pos):
    statement = lines[start_line_idx][start_pos:]
    
    open_parens = statement.count('(') - statement.count(')')
    line_idx = start_line_idx
    
    while (open_parens > 0 or statement.rstrip().endswith('\\') or 
           statement.rstrip().endswith(',')) and line_idx < len(lines) - 1:
        line_idx += 1
        next_line = lines[line_idx]
        statement += '\n' + next_line
        open_parens += next_line.count('(') - next_line.count(')')
    
    return statement, start_line_idx, line_idx


def analyze_logging_statements(file_path):
    """
    Analyze a Python file for logging statements and return them with line numbers.
    
    Args:
        file_path: Path to the Python file to analyze
        
    Returns:
        List of tuples containing (line_number, logging_statement)
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            
        logging_statements = []
        
        # Regular expressions for different types of logging statements
        patterns = [
            # r'([\w\.]+\.add_(scalar|histogram|image|text|graph)\(.*?\))',
              
            # r'([\w\.]+\.?log(?:_\w+)?\((?:[^)(]|\([^)(]*\))*\))', #common
            
            # r'([\w\.]+\.?metadata(?:_\w+)?\((?:[^)(]|\([^)(]*\))*\))',

            # r'[\w\.]+\.log_\w+\s*\(.*', # xxx.xxx.log_xxx(xxx)
            # r'[\w\.]+\.log\(.*', # xxx.xxx.log(xxx)
            # r'[\w\.]+\.watch\(.*', # xxx.xxx.watch(xxx)
            # r'[\w\.]+\.fit\(.*', # xxx.xxx.fit(xxx)
            # r'[\w\.]+(?:\["[\w_]+"\])?\.add_\w+\(.*', # xxx.xxx(["xxx"]).add_xxx(xxx
            # r'[\w\.]+\.report_\w+\(.*', # xxx.xxx.report_xxx(xxx)
            # r'[\w\.]+\.attach\(.*', # xxx.xxx.attach(xxx)
            # r'[\w\.]+\.metadata\(.*', # xxx.xxx.metadata(xxx)
            # r'[\w\.]+\.event\(.*', # xxx.xxx.event(xxx)
            # r'[\w\.]+(logger|logging)\.?\w+\(.*', # xxx.xxx.logger/logging.xxx(xxx
            # r'[\w\.]+summary\.?\w+\(.*'
            r'(logger|logging)\.?\w+\(.*', # xxx.xxx.logger/logging.xxx(xxx


            
        ]

        matched_positions = set()

        for i, line in enumerate(lines):
            for pattern in patterns:
                for match in re.finditer(pattern, line):
                    start_pos = match.start()
                    position_key = (i, start_pos)
                    if position_key in matched_positions:
                        continue
                    matched_positions.add(position_key)
                    full_statement, start_line, end_line = get_full_logging_statement(lines, i, start_pos)
                    logging_statements.append((start_line + 1, full_statement.strip()))
                
     
        return logging_statements
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

def find_logging_calls(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        logging_calls = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and 
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == 'logging' or node.func.value.id == 'logger'):
                    logging_calls.append((node.lineno, ast.unparse(node)))
            
                
        return logging_calls
    except SyntaxError:

        return analyze_logging_statements(content)

def main():
    parent_directory = "../dataset"
    project_folders = [
        folder for folder in os.listdir(parent_directory)
        if os.path.isdir(os.path.join(parent_directory, folder))
    ]
    
    output_dir = Path("logging_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for project in project_folders:
        project_path = os.path.join(parent_directory, project)

        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}...")
                    logging_statements = analyze_logging_statements(file_path)  
                    if logging_statements:
                        results.append({
                            "file_path": file_path,
                            "logging_statements": logging_statements,
                            "statement_count": len(logging_statements)
                        })              
    
    # Write results to CSV
    results_df = pd.DataFrame([
        {
            "file_path": r["file_path"],
            "ml_logging_details": r["ml_logging_details"],
            "narrowed_ml_logging_details": r["narrowed_ml_logging_details"],
            "logging_statements": r["logging_statements"],
            "statement_count": r["statement_count"]
        } for r in results
    ])
    
    results_df.to_csv(output_dir / "logging_summary.csv", index=False)
    
if __name__ == "__main__":
    main()
