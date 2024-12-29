from pathlib import Path
from typing import List

import yaml

from inspect_ai import Task
from .question import Question


class Benchmark:
    """A collection of questions to evaluate.
    
    This class is responsible for loading and validating the questions
    """

    def __init__(self, questions: List[Question]):
        self.questions = questions
        self._validate()

    def build_tasks(self) -> List[Task]:
        """Build tasks from the questions."""
        return [q.build_task() for q in self.questions]
        
    def _validate(self) -> None:
        """Validate the entire configuration."""
        # Check for duplicate IDs
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("Question IDs must be unique")
            
        # Validate each question
        for question in self.questions:
            question.validate()

    def get_question_by_id(self, id: str) -> Question:
        """Get a question by its ID."""
        for question in self.questions:
            if question.id == id:
                return question
        raise ValueError(f"Question with ID {id} not found")
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'Benchmark':
        """Load configuration from a YAML file."""
        with open(path) as f:
            raw_config = yaml.safe_load(f)
            
        questions = [Question(**q) for q in raw_config]
        return cls(questions)
    
    @classmethod
    def from_yaml_dir(cls, dir_path: Path) -> 'Benchmark':
        """Load and merge configurations from all YAML files in a directory."""
        questions = []
        for yaml_file in dir_path.glob("*.yaml"):
            with open(yaml_file) as f:
                raw_config = yaml.safe_load(f)
                questions.extend([Question(**q) for q in raw_config])
                
        return cls(questions)