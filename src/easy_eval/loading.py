import yaml

from pathlib import Path
from .question import Question, QuestionConfig


def load_question_from_yaml(id: str, path: Path) -> Question:
    """Load a specific question from a YAML file.
    
    Args:
        id: ID of the question to load
        path: Path to YAML file containing question configurations
        
    Returns:
        Question object with matching ID from the YAML file
        
    Raises:
        ValueError: If no question with matching ID is found
    """
    with open(path) as f:
        raw_config = yaml.safe_load(f)
        
    questions = [Question(QuestionConfig(**q)) for q in raw_config if q.get('id') == id]
    if len(questions) > 1:
        raise ValueError(f"Multiple questions found with id '{id}' in {path}")
    
    if not questions:
        raise ValueError(f"No question found with id '{id}' in {path}")
        
    return questions[0]

def load_question_from_yaml_dir(id: str, dir_path: Path) -> Question:
    """Load a specific question from all YAML files in a directory.
    
    Args:
        id: ID of the question to load
        dir_path: Path to directory containing YAML files
        
    Returns:
        Question object with matching ID from the YAML files
        
    Raises:
        ValueError: If no question with matching ID is found
    """
    for yaml_file in dir_path.glob("*.yaml"):
        try:
            return load_question_from_yaml(id, yaml_file)
        except ValueError:
            continue
            
    raise ValueError(f"No question found with id '{id}' in directory {dir_path}")