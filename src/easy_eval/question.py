import hashlib
import json

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from inspect_ai.dataset import Sample
from inspect_ai.solver import Solver, generate, system_message
from inspect_ai.scorer import Scorer
from inspect_ai import Task, task

from easy_eval.scorer import dummy, model_graded_rating

QuestionMetadata = dict[str, str]
QuestionType = Literal[
    "free_form",
    "answer_0_100", # The model is supposed to answer with a number between 0 and 100
    "judge_0_100", # A judge model will grade the model's answer between 0 and 100
]

@dataclass
class Question:
    id: str
    type: QuestionType
    paraphrases: List[str]
    samples_per_paraphrase: int
    target: Optional[str] = None
    system_prompt: Optional[str] = None
    judge_models: Optional[str | list[str]] = None
    judge_prompts: Optional[Dict[str, str]] = None

    def validate(self) -> None:
        """Validate the question configuration."""
        if not self.id:
            raise ValueError("Question ID cannot be empty")
        
        if not self.paraphrases:
            raise ValueError(f"Question {self.id}: must have at least one paraphrase")
            
        if self.samples_per_paraphrase < 1:
            raise ValueError(f"Question {self.id}: samples_per_paraphrase must be positive")
            
        if self.type == "judge_0_100":
            if not self.judge_models:
                raise ValueError(f"Question {self.id}: judge model required for {self.type}")
            if not self.judge_prompts:
                raise ValueError(f"Question {self.id}: judge_prompts required for {self.type}")

    def hash(self) -> str:
        """Hash the question configuration."""
        # Convert dataclass to dictionary using all attributes
        config_dict = {
            'id': self.id,
            'type': self.type,
            'paraphrases': self.paraphrases,
            'samples_per_paraphrase': self.samples_per_paraphrase,
            'target': self.target,
            'system_prompt': self.system_prompt,
            'judge_models': self.judge_models,
            'judge_prompts': self.judge_prompts
        }
        return hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()

    def build_task(self) -> Task:
        """Build a Task from this Question."""
        # Call the task decorator in order to register the task
        @task(name = self.id)
        def _task_fn():
            return Task(
                dataset=self.build_dataset(),
                solver=self.build_solver(),
                scorer=self.build_scorer()
            )

        return _task_fn()

    # TODO: change build_dataset, build_scorer to be private methods? 

    def build_dataset(self) -> List[Sample]:
        """Build a dataset from this Question."""
        samples = []
        
        for paraphrase_idx, paraphrase in enumerate(self.paraphrases):
            for sample_idx in range(self.samples_per_paraphrase):
                metadata: QuestionMetadata = {
                    "question_id": self.id,
                    "question_type": self.type,
                    "paraphrase_index": paraphrase_idx,
                    "sample_index": sample_idx,
                    "judge_models": self.judge_models,
                    "judge_prompts": self.judge_prompts
                }            

                # Create unique ID for each sample
                sample_id = f"{self.id}_p{paraphrase_idx}_s{sample_idx}"

                target = self.target or ""

                # Generate a sample                
                sample = Sample(
                    input=paraphrase,  # Assuming string input, not ChatMessage
                    id=sample_id,
                    target=target,
                    metadata=metadata
                )
                samples.append(sample)
        
        return samples
    
    def build_solver(self) -> list[Solver]:
        """Build a solver for this Question."""
        solver = []
        if self.system_prompt:
            solver.append(system_message(self.system_prompt))
        solver.append(generate())
        return solver
    
    def build_scorer(self) -> list[Scorer]:
        """Build a scorer for this Question."""
        if self.type == "judge_0_100":
            scorers = []
            for judge_metric, judge_prompt in self.judge_prompts.items():
                # Ugh moment: Can't name the scorer based on runtime variables
                # This is because we add the scorer to registry before we know the name
                # TODO: How to fix this? 

                # Potential fix: 
                # - Define runtime lambda functions as scorers which wrap the model_graded_rating
                # - Then, name the scorer based on the lambda function
                scorers.append(model_graded_rating(
                    model=self.judge_models,
                    criterion=judge_prompt,
                ))
            return scorers
        elif self.type == "answer_0_100":
            raise NotImplementedError("Free form 0-100 scoring not implemented")
        elif self.type == "free_form":
            # For free-form questions, we just want to collect responses
            # Return a scorer that always gives a score of 1
            return [dummy()]
        else:
            raise ValueError(f"Unsupported question type: {self.type}")