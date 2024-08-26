import os
from pathlib import Path
from typing import Dict, List, Tuple

def create_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {path}")

def create_file(path: Path, content: str = "") -> None:
    path.write_text(content)
    print(f"Created file: {path}")

def create_project_structure() -> None:
    # Create subdirectories
    directories: List[str] = [
        ".github/workflows",
        "rag_gemma/api",
        "rag_gemma/frontend",
        "rag_gemma/model",
        "rag_gemma/scripts",
        "data/raw",
        "data/processed",
        "tests",
        "local_models",
        "saved_models",
    ]
    
    for directory in directories:
        create_directory(Path(directory))
    
    # Create __init__.py files
    init_files: List[str] = [
        "rag_gemma/__init__.py",
        "rag_gemma/api/__init__.py",
        "rag_gemma/frontend/__init__.py",
        "rag_gemma/model/__init__.py",
        "rag_gemma/scripts/__init__.py",
        "tests/__init__.py",
    ]
    
    for init_file in init_files:
        create_file(Path(init_file))
    
    # Create other necessary files
    files: List[Tuple[str, str]] = [
        (".github/workflows/ci.yml", "# TODO: Add CI workflow"),
        (".github/workflows/cd.yml", "# TODO: Add CD workflow"),
        ("rag_gemma/api/main.py", "# TODO: Implement FastAPI app"),
        ("rag_gemma/api/requirements.txt", "fastapi\nuvicorn"),
        ("rag_gemma/frontend/streamlit_app.py", "# TODO: Implement Streamlit app"),
        ("rag_gemma/frontend/requirements.txt", "streamlit"),
        ("rag_gemma/model/gemma_model.py", "# TODO: Implement Gemma model"),
        ("rag_gemma/model/requirements.txt", "transformers\ntorch\nsentence-transformers"),
        ("rag_gemma/scripts/preprocess.py", "# TODO: Implement data preprocessing"),
        ("rag_gemma/scripts/train.py", "# TODO: Implement model training"),
        ("rag_gemma/scripts/retrain.py", "# TODO: Implement model retraining"),
        ("tests/test_api.py", "# TODO: Implement API tests"),
        ("tests/test_model.py", "# TODO: Implement model tests"),
        (".gitignore", "# Python\n__pycache__/\n*.pyc\n\n# Virtual environment\nvenv/\n\n# Data\ndata/"),
        ("docker-compose.yml", "# TODO: Define services"),
        ("README.md", "# RAG Gemma Project\n\nA RAG system using Gemma 2B model."),
        ("requirements.txt", "# TODO: Add project dependencies"),
        ("setup.py", "# TODO: Define package setup"),
        ("Dockerfile", "# TODO: Define Dockerfile"),
        (".env", "# TODO: Define environment variables"),
    ]
    
    for file_path, content in files:
        create_file(Path(file_path), content)

def main() -> None:
    create_project_structure()
    print("Project structure created successfully!")

if __name__ == "__main__":
    main()