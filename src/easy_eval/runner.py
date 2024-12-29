import hashlib
from pathlib import Path

from inspect_ai import eval
from inspect_ai.log import EvalLog, write_eval_log, read_eval_log
from easy_eval.benchmark import Benchmark

def get_filename(question_hash: str, model_hash: str) -> Path:
    return hashlib.sha256(f"{question_hash}_{model_hash}".encode()).hexdigest()

class Runner:

    log_dir: Path

    def __init__(self, log_dir: str | Path = "./logs"):
        self.log_dir = Path(log_dir)
        self.inspect_log_dir = self.log_dir.parent / "_inspect_logs"

    def run(self, benchmark: Benchmark, models: list[str]):
        """Run the benchmark on a given set of models."""
        tasks = benchmark.build_tasks()
        # TODO: Filter tasks that have already been run

        # Save the inspect logs somewhere else
        logs: list[EvalLog] = eval(tasks = tasks, model = models, log_dir = str(self.inspect_log_dir))

        # Ugh moment: `eval` doesn't check if the task has been run before 
        # We'd like to check this and skip tasks that have already been run
        # Ugh moment: `eval` saves to a custom log directory that is kind of inscrutable
        # We'd like to save to a more user-friendly directory

        # Potential fix to both of the above: 
        # - Write logs to a custom directory, with hash determined based on the question config
        # - Check if the task has been run before by checking the log directory
        # - This should be doable by just calling `write_eval_log` 

        # TODO: Implement this, preferably in a new Runner class that wraps eval
        for log in logs:
            question_id = log.eval.task.split("/")[-1]
            question_hash = benchmark.get_question_by_id(question_id).hash()
            model_hash = log.eval.model
            # Filename is a hash of the question and model
            # TODO: Are there other relevant variables to include in the hash?
            log_path = self.log_dir / get_filename(question_hash, model_hash)
            log_path = log_path.with_suffix(".eval")
            # Skip the failed logs
            if log.status == "success":
                write_eval_log(log, str(log_path), format="eval")
            else: 
                print(f"Skipping {log_path} because it failed")

    def load_logs(self):
        """Load the results from the log directory."""
        logs = []
        for log_path in self.log_dir.glob("*.eval"):
            log = read_eval_log(str(log_path))
            if log.status == "success":
                logs.append(log)
        return logs
