import os
from pathlib import Path

# --- Configuration ---
# The root directory of your project
START_PATH = Path('.')
# Directories to completely ignore
IGNORE_DIRS = {'.git', '.vscode', '__pycache__', '.venv', 'node_modules', 'dist', 'build'}
# --- End Configuration ---

def generate_tree(dir_path: Path, prefix: str = ''):
    """Recursively generates a visual tree structure."""
    
    # Get contents, filter ignored, and separate dirs and files
    contents = [p for p in dir_path.iterdir() if p.name not in IGNORE_DIRS]
    dirs = sorted([p for p in contents if p.is_dir()])
    files = sorted([p for p in contents if p.is_file()])
    
    # Combine them for printing
    entries = dirs + files
    
    for i, entry in enumerate(entries):
        # Determine the connector for the tree
        connector = '├── ' if i < len(entries) - 1 else '└── '
        print(f'{prefix}{connector}{entry.name}')
        
        # If it's a directory, recurse into it
        if entry.is_dir():
            extension = '│   ' if i < len(entries) - 1 else '    '
            generate_tree(entry, prefix=prefix + extension)

if __name__ == '__main__':
    print(f'{START_PATH.name}/')
    generate_tree(START_PATH)