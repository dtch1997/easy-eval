import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pathlib import Path
from lm_query import Benchmark, Runner
from dotenv import load_dotenv
from inspect_ai.log import EvalLog

curr_dir = Path(__file__).parent

# Load env variables
load_dotenv(curr_dir / ".env")
# Load benchmark from YAML files 
benchmark = Benchmark.from_yaml_dir(curr_dir)
# Define models to evaluate
models = [
    # OpenAI models
    "openai/gpt-4o-mini-2024-07-18",
    # "openai/gpt-4o-2024-08-06",

    # Anthropic models
    "anthropic/claude-3-5-haiku-20241022",
    # "anthropic/claude-3-5-sonnet-20241022"
]

def parse_results(logs: list[EvalLog]) -> pd.DataFrame:
    rows = []
    for log in logs:

        # Get the task and model
        question_id = log.eval.task.split("/")[-1]
        model = log.eval.model

        # Get the metrics
        scores = log.results.scores
        assert len(scores) == 1  # Only one scorer per task
        score = scores[0]
        metrics = score.metrics

        row = {
            "question_id": question_id,
            "model": model,
            # TODO: support groupings of models
        }
        for metric_name, metric in metrics.items():
            row[metric_name] = metric.value
        rows.append(row)

    df = pd.DataFrame(rows)

    return df

if __name__ == "__main__":
    print(models)
    runner = Runner(log_dir=curr_dir / "logs")
    runner.run(benchmark, models)
    logs = runner.load_logs()

    # Plot the results
    df = parse_results(logs)
    print(df)

    # Plot the results
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="model", y="mean", data=df)
    plt.show()
