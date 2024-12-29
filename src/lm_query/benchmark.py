from pathlib import Path
from typing import List

import yaml

from .question import Question
from inspect_ai import eval


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
        # Ugh moment: `eval` doesn't check if the task has been run before 
        # We'd like to check this and skip tasks that have already been run
        # Ugh moment: `eval` saves to a custom log directory that is kind of inscrutable
        # We'd like to save to a more user-friendly directory

        # Potential fix to both of the above: 
        # - Write logs to a custom directory, with hash determined based on the question config
        # - Check if the task has been run before by checking the log directory
        # - This should be doable by just calling `write_eval_log` 

        # TODO: Implement this, preferably in a new Runner class that wraps eval