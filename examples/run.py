from pathlib import Path
from inspect_ai import eval
from lm_query import Benchmark
from dotenv import load_dotenv

curr_dir = Path(__file__).parent

# Load the configuration
load_dotenv(curr_dir / ".env")
config = Benchmark.from_yaml_dir(curr_dir)

models = [
    # OpenAI models
    "openai/gpt-4o-2024-08-06",
    # Anthropic models
    "anthropic/claude-3-5-sonnet-20241022"
]

# Convert each question to a task and evaluate
tasks = [q.build_task() for q in config.questions]
logs = eval(tasks = tasks, model = models)

# Results will contain:
# - For free_form_0_100: Direct numerical scores
# - For free_form_judge_0_100: Aggregated judge scores per criterion
# - For free_form: Raw responses for qualitative analysis