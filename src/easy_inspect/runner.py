import hashlib
import pandas as pd

from pathlib import Path
from inspect_ai import eval
from inspect_ai.log import EvalLog, write_eval_log, read_eval_log
from easy_inspect.question import Question

def get_filename(question_hash: str | int, model_hash: str | int) -> Path:
    return hashlib.sha256(f"{question_hash}_{model_hash}".encode()).hexdigest()

class Runner:

    log_dir: Path
    question: Question | None = None
    models: list[str] | None = None

    def __init__(self, log_dir: str | Path = "./logs"):
        self.log_dir = Path(log_dir)
        # Put the base inspect logs in a subdirectory
        self.inspect_log_dir = self.log_dir / "inspect_logs"

    # Use builder pattern to set the question and models

    def with_question(self, question: Question):
        self.question = question
        return self

    def with_models(self, models: list[str]):
        self.models = models
        return self

    def run(self):
        """Run the question on a given set of models."""

        if not self.question:
            raise ValueError("Question not set")
        if not self.models:
            raise ValueError("Models not set")

        task = self.question.build_task()
        # TODO: Filter tasks that have already been run

        # Save the inspect logs somewhere else
        logs: list[EvalLog] = eval(tasks = [task], model = self.models, log_dir = str(self.inspect_log_dir))

        # Motivation for the subsequent code:
        # - inspect_ai's `eval` function doesn't allow caching previous runs. 
        # - We'd like to be able skip tasks that have already been run.

        # Implemented fix: 
        # - Write logs to a custom directory, with hash determined based on the question config
        # - Check if the task has been run before by checking the log directory

        for log in logs:
            question_hash = self.question.hash()
            model_hash = log.eval.model
            # New filename is a hash of the question and model
            # TODO: Are there other relevant variables to include in the hash?
            log_path = self.log_dir / get_filename(question_hash, model_hash)
            log_path = log_path.with_suffix(".eval")
            # Skip the failed logs
            if log.status == "success":
                write_eval_log(log, str(log_path), format="eval")
            else: 
                print(f"Skipping {log_path} because it failed")

    def load_logs(self) -> list[EvalLog]:
        """Load the results from the log directory."""
        logs = []
        for log_path in self.log_dir.glob("*.eval"):
            log = read_eval_log(str(log_path))
            if log.status != "success":
                continue
            # Hacky way to check if the log is for the current question
            if log.eval.task.split("/")[-1] == self.question.config.id:
                logs.append(log)
        return logs

    def parse_results(self, logs: list[EvalLog]) -> pd.DataFrame:
        """Parse the results from the logs into a DataFrame."""
        # NOTE: Parsing logic handles some quirks of the `run` method 
        # Hence why they are bundled together in the `Runner` class
        rows = []
        for log in logs:

            # Get the task and model
            question_id = log.eval.task.split("/")[-1]
            model = log.eval.model

            row = {
                "question_id": question_id,
                "model": model,
                # TODO: support groupings of models
            }

            # Get the metrics
            scores = log.results.scores
            for score in scores:
                # get the scorer name
                if "name" in score.params:
                    # If name was set at runtime, use that instead
                    name = score.params["name"]
                else:
                    # Else, default to the scorer name
                    name = score.name

                metrics = score.metrics
                for metric_name, metric in metrics.items():
                    row[f"{name}/{metric_name}"] = metric.value

            rows.append(row)

        df = pd.DataFrame(rows)
        return df
    
    def load_results(self) -> pd.DataFrame:
        """Load the results from the log directory.
        
        Syntactic sugar for `load_logs` and `parse_results`.
        """
        logs = self.load_logs()
        return self.parse_results(logs)
