import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pathlib import Path
from lm_query import Benchmark
from dotenv import load_dotenv
from inspect_ai.log import read_eval_log

curr_dir = Path(__file__).parent

# Load env variables
load_dotenv(curr_dir / ".env")
# Load benchmark from YAML files 
benchmark = Benchmark.from_yaml_dir(curr_dir)
# Define models to evaluate
models = [
    # OpenAI models
    "openai/gpt-4o-2024-08-06",
    # Anthropic models
    "anthropic/claude-3-5-sonnet-20241022"
]

def load_results(results_dir: Path) -> pd.DataFrame:
    result_files = list(results_dir.glob("*.eval"))
    rows = []

    for result_file in result_files:
        print(f"Processing {result_file.name}")
        log = read_eval_log(str(result_file))
        if not log.status == "success":
            print(f"Skipping {result_file.name} because it failed")
            continue

        # Get the task and model
        question_id = log.eval.task
        model = log.eval.model

        # Get the metrics
        scores = log.results.scores
        print(scores)
        # assert len(scores) == 1  # Only one scorer per task
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
    benchmark.run(models)

    # Plot the results
    results_dir = curr_dir / "logs"
    df = load_results(results_dir)
    print(df)

    # Plot the results
    sns.set_theme(style="whitegrid")
    sns.barplot(x="model", y="mean", data=df)
    plt.show()
