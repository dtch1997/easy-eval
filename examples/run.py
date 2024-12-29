from pathlib import Path
from easy_eval import Runner
from easy_eval.loading import load_question_from_yaml_dir
from easy_eval.plotting import models_plot
from dotenv import load_dotenv

curr_dir = Path(__file__).parent

# Load env variables
load_dotenv(curr_dir / ".env")



if __name__ == "__main__":
    models = [
        # OpenAI models
        "openai/gpt-4o-mini-2024-07-18",
        # "openai/gpt-4o-2024-08-06",

        # Anthropic models
        "anthropic/claude-3-5-haiku-20241022",
        # "anthropic/claude-3-5-sonnet-20241022"
    ]

    runner = Runner(log_dir=curr_dir / "logs")

    # Example 1: Free-form question
    question = load_question_from_yaml_dir("example_1", curr_dir)
    runner.with_question(question).with_models(models).run()
    df = runner.load_results()
    print(df)

    # Example 2: Free-form question, judged on 0-100 scale
    question = load_question_from_yaml_dir("example_2", curr_dir)
    runner.with_question(question).with_models(models).run()
    df = runner.load_results()
    print(df)

    # Plot the results
    # TODO: support error bars
    models_plot(df, metric="ethical_reasoning/mean")
    models_plot(df, metric="harm_consideration/mean")
