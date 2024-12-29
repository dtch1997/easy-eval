from pathlib import Path
from lm_query import Benchmark
from dotenv import load_dotenv

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

if __name__ == "__main__":
    benchmark.run(models)