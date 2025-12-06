# src/utils.py

EVALUATION_PROMPT = """Please act as an impartial judge and evaluate the text quality of the provided passage. Text quality of a passage is defined by how well it maintains the following aspects.

(1) Coherence - logical flow and connectivity between sentences and ideas in the text.
(2) Fluency - smoothness and naturalness of individual sentences.
(3) Simplicity - how easy it is to understand the passage.
(4) Linguistic Acceptability - if the text sounds natural and correct to a native speaker.

Provide a binary rating, “0” for low quality or “1" for high quality, strictly following this format: “[[0]]” or “[[1]]“. Do not provide an explanation.
Passage:
{passage}
"""

def format_dpo_entry(passage, rating):
    """
    Formats the conversation for the model.
    """
    user_content = EVALUATION_PROMPT.format(passage=passage)
    assistant_content = f"Rating: [[{rating}]]"
    
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
