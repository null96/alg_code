"""
openclaw_task - Entry point for relay worker execution.
Command: python -m openclaw_task [args...]
"""
import sys
import os
import subprocess

# Add the repo root to path so automl_pipeline is importable
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Install requirements if needed
req_file = os.path.join(repo_root, 'automl_pipeline', 'requirements.txt')
if os.path.exists(req_file):
    print("Installing requirements...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', '-r', req_file])
    print("Requirements installed.")

# Strip the first arg if it's the module name (relay passes it)
if len(sys.argv) > 1 and sys.argv[1] in ('automl_pipeline', 'openclaw_task'):
    sys.argv = sys.argv[1:]

# Import and run the automl pipeline
from automl_pipeline.__main__ import main

if __name__ == '__main__':
    sys.exit(main())
