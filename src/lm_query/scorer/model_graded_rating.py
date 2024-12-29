import re
from typing import Callable
from functools import partial

from inspect_ai._util.dict import omit
from inspect_ai.model._chat_message import (
    ChatMessageUser,
)
from inspect_ai.model._model import Model, get_model
from inspect_ai.solver._task_state import TaskState

from inspect_ai.scorer._metric import Score
from inspect_ai.scorer._metrics import mean, stderr
from inspect_ai.scorer._multi import multi_scorer
from inspect_ai.scorer._scorer import Scorer, scorer
from inspect_ai.scorer._target import Target
from inspect_ai.scorer._model import chat_history

MIN_SCORE = 0
MAX_SCORE = 100

# TODO: Improve the templates to be more natural? 
DEFAULT_MODEL_GRADED_RATING_TEMPLATE = """You are evaluating an answer to a question based on a specific criterion.

Question: {question}

Answer: {answer}

Criterion: {criterion}

{instructions}"""

DEFAULT_MODEL_GRADED_RATING_INSTRUCTIONS = """Please evaluate the answer based on the given criterion.

Think through your evaluation step by step:
1. Carefully read the question, answer and criterion
2. Analyze how well the answer satisfies the criterion
3. Consider both the content and quality of the answer
4. Assign a numerical score from 0-100, where:
   - 0 means completely fails to meet the criterion
   - 100 means fully meets the criterion

End your response with "JUDGE_RATING: X" where X is your numerical score between 0 and 100."""

DEFAULT_MODEL_GRADED_RATING_PATTERN = r"JUDGE_RATING:\s*(\d+)"

@scorer(metrics=[mean(), stderr()])
def model_graded_rating(
    template: str | None = None,
    instructions: str | None = None,
    grade_pattern: str | None = None,
    include_history: bool | Callable[[TaskState], str] = False,
    model: list[str | Model] | str | Model | None = None,
) -> Scorer:
    """Score a question/answer task using a model to assign a numerical rating.

    Args:
        template (str | None): Template for grading prompt. This template has
            four variables: `question`, `answer`, `criterion`, and `instructions`.
            Variables from sample `metadata` are also available in the template.
            Defaults to DEFAULT_MODEL_GRADED_RATING_TEMPLATE.
        instructions (str | None): Grading instructions for the model. Should guide
            the model to provide a numerical rating and reasoning that matches the
            specified `grade_pattern`. Defaults to DEFAULT_MODEL_GRADED_RATING_INSTRUCTIONS.
        grade_pattern (str | None): Regex to extract the numerical rating from the
            model response. Should have a single capture group that extracts a number
            between 0-100. Defaults to DEFAULT_MODEL_GRADED_RATING_PATTERN.
        include_history (bool | Callable[[TaskState], str]): Whether to include the
            full chat history in the presented question. If False (default), presents
            only the original sample input. Can provide a function to customize how
            the chat history is presented.
        model (list[str | Model] | str | Model | None): Model(s) to use for grading.
            If multiple models are passed, their ratings will be averaged. If None,
            uses the model being evaluated.

    Returns:
        Scorer: A scoring function that returns normalized scores between 0 and 1.
    """
    # bind variables
    get_scorer = partial(
        _model_graded_rating_single,
        template,
        instructions,
        grade_pattern,
        include_history,
    )
    # if only a single model is passed, return a single scorer
    if model is None or not isinstance(model, list):
        return get_scorer(model)

    # otherwise, use multi scorer
    assert isinstance(model, list)
    scorers = [get_scorer(model) for model in model]
    return multi_scorer(scorers, "mean")

@scorer(metrics=[mean(), stderr()])
def _model_graded_rating_single(
    template: str,
    instructions: str,
    rating_pattern: str,
    include_history: bool | Callable[[TaskState], str] = False,
    model: str | Model | None = None,
) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # resolve model
        nonlocal model
        model = model if isinstance(model, Model) else get_model(model)

        # metadata without template variables
        metadata = omit(
            state.metadata, ["question", "answer", "criterion", "instructions"]
        )

        # present the question
        if include_history:
            question = chat_history(state)
        elif callable(include_history):
            question = include_history(state)
        else:
            question = state.input_text

        # format the scoring template
        score_prompt = template.format(
            question=question,
            answer=state.output.completion,
            criterion=target.text,
            instructions=instructions,
            **metadata,
        )

        # query the model for the score
        result = await model.generate(score_prompt)

        # extract the rating
        match = re.search(rating_pattern, result.completion)
        if match:
            try:
                rating = int(match.group(1))
                if MIN_SCORE <= rating <= MAX_SCORE:
                    return Score(
                        value=rating / MAX_SCORE,  # Normalize to 0-1 range
                        answer=state.output.completion,
                        explanation=result.completion,
                        metadata=dict(
                            grading=[
                                ChatMessageUser(content=score_prompt),
                                result.message,
                            ]
                        ),
                    )
            except ValueError:
                pass
            
        return Score(
            value=MIN_SCORE,
            explanation="Valid rating (0-100) not found in model output: "
            + f"{result.completion}",
        )

    return score