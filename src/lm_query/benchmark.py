from pathlib import Path
from typing import List

import yaml

from .question import Question


class Benchmark:
    def __init__(self, questions: List[Question]):
        self.questions = questions
        self._validate()
        
    def _validate(self) -> None:
        """Validate the entire configuration."""
        # Check for duplicate IDs
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("Question IDs must be unique")
            
        # Validate each question
        for question in self.questions:
            question.validate()
    
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
    
    def run(self, model: str | list[str]) -> None:
        """Run the benchmark on a given model."""
        tasks = [q.build_task() for q in self.questions]
        eval(tasks = tasks, model = model)