from inspect_ai.scorer import Scorer, Score, scorer

@scorer(metrics=[])
def dummy() -> Scorer:
    async def score_fn(state, target):
        del state, target
        return Score(
            value=1.0,
            explanation="Dummy scorer does not require scoring. "
            "All responses are given a score of 1."
        )
    return score_fn