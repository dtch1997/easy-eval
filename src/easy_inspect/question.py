import hashlib
import json

from dataclasses import dataclass
from typing import List, Literal, Optional

from inspect_ai.dataset import Sample
from inspect_ai.solver import Solver, generate, system_message
from inspect_ai.scorer import Scorer
from inspect_ai import Task, task

from easy_inspect.scorer import dummy, model_graded_rating

QuestionMetadata = dict[str, str]
QuestionType = Literal[
    "free_form",
    "free_form_judge_0_100",  # A judge model will grade the model's answer between 0 and 100
    "answer_0_100", # The model is supposed to answer with a number between 0 and 100
    "free_form_judge", # A judge model will grade the model's answer in an arbitrary way
]

@dataclass(frozen=True)
class QuestionConfig:
    id: str
    type: QuestionType
    paraphrases: list[str]
    samples_per_paraphrase: int
    target: Optional[str] = None    
    system_prompt: Optional[str] = None
    judge_models: Optional[str | list[str]] = None
    judge_prompts: Optional[list[dict[str, str]]] = None

    def validate(self) -> None:
        """Validate the question configuration."""
        if not self.id:
            raise ValueError("Question ID cannot be empty")
        
        if not self.paraphrases:
            raise ValueError(f"Question {self.id}: must have at least one paraphrase")
            
        if self.samples_per_paraphrase < 1:
            raise ValueError(f"Question {self.id}: samples_per_paraphrase must be positive")
            
        if self.type == "free_form_judge_0_100":
            if not self.judge_models:
                raise ValueError(f"Question {self.id}: judge model required for {self.type}")
            if not self.judge_prompts:
                raise ValueError(f"Question {self.id}: judge_prompts required for {self.type}")

    def hash(self) -> str:
        """This is a unique identifier of a question. Changes when we change the wording.
        
        We use that to determine whether we can use cached results.
        """
        attributes = {k: v for k, v in self.__dict__.items()}
        json_str = json.dumps(attributes, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

class Question:
    def __init__(self, config: QuestionConfig):
        self.config = config
        self.config.validate()

    def hash(self) -> str:
        return self.config.hash()

    def build_task(self) -> Task:
        """Build a Task from this Question."""
        # Call the task decorator in order to register the task
        @task(name = self.config.id)
        def _task_fn():
            return Task(
                dataset=self.build_dataset(),
                solver=self.build_solver(),
                scorer=self.build_scorer(),
            )

        return _task_fn()

    # TODO: change build_dataset, build_scorer to be private methods? 

    def build_dataset(self) -> List[Sample]:
        """Build a dataset from this Question."""
        samples = []
        
        for paraphrase_idx, paraphrase in enumerate(self.config.paraphrases):
            for sample_idx in range(self.config.samples_per_paraphrase):
                metadata: QuestionMetadata = {
                    "question_id": self.config.id,
                    "question_type": self.config.type,
                    "paraphrase_index": paraphrase_idx,
                    "sample_index": sample_idx,
                    "judge_models": self.config.judge_models,
                    "judge_prompts": self.config.judge_prompts
                }            

                # Create unique ID for each sample
                sample_id = f"{self.config.id}_p{paraphrase_idx}_s{sample_idx}"

                target = self.config.target or ""

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
        if self.config.system_prompt:
            solver.append(system_message(self.config.system_prompt))
        solver.append(generate())
        return solver
    
    def build_scorer(self) -> list[Scorer]:
        """Build a scorer for this Question."""
        if self.config.type == "free_form_judge_0_100":
            scorers = []
            for name, prompt in self.config.judge_prompts.items():
                scorers.append(model_graded_rating(
                    name=name,
                    model=self.config.judge_models,
                    criterion=prompt,
                ))
            return scorers
        elif self.config.type == "free_form":
            # For free-form questions, we just want to collect responses
            # Return a scorer that always gives a score of 1
            return [dummy()]
        elif self.config.type == "answer_0_100":
            raise NotImplementedError("Free form 0-100 scoring not implemented")
        elif self.config.type == "free_form_judge":
            raise NotImplementedError("Free form judge not implemented")
        else:
            raise ValueError(f"Unsupported question type: {self.config.type}")
