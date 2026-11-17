"""Fixed IDK, granularity-aware and DRAG baseline instructions."""

GRANULARITY_INSTRUCTION = (
    "Answer at a level of detail you are confident is correct. "
    "If you are uncertain about finer details, give a broader answer rather "
    "than guessing. If you cannot give a confident answer even at a broader "
    'level, say "I don\'t know."'
)


IDK_USER_PROMPT_TEMPLATE = (
    "You will be given a question.  Answer the question, or, if you are not "
    "certain of the answer, output IDK. Question: {question} Answer:"
)


DRAG_AGGREGATION_INSTRUCTION = """You will be given a list of responses; replace them with the most specific answer that is still consistent with all the original responses. If the responses have nothing meaningful in common with respect to the question, output IDK.

Here are some examples:

Question: Where was [X] born?
Responses:
- Hamburg
- Hamburg
- Bonn
- Berlin
Correct aggregated answer: Germany
Incorrect aggregated answer: Hamburg
Explanation: These are all different cities in Germany. Hamburg is not a correct aggregation, since it is not consistent with other responses, such as Berlin or Bonn.

Question: When was [X] born?
Responses:
- February 1, 1937
- November 20, 1937
- January 1937
Correct aggregated answer: 1937
Incorrect aggregated answer: November 1937
Explanation: These are all dates in 1937."""


def build_drag_aggregation_prompt(
    question: str,
    responses: list[str] | tuple[str, ...],
) -> str:
    """Build the legacy DRAG prompt without changing response text."""
    if not responses:
        raise ValueError("DRAG requires at least one sampled response")
    return (
        f"{DRAG_AGGREGATION_INSTRUCTION}\n\n"
        f"Question: {question}\n"
        "Responses:\n"
        + "\n".join(f"- {response}" for response in responses)
        + "\nAggregated answer:"
    )

