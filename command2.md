Task: Agent Dictionary Expansion

Hi Gaius,

For this task the focus is structure, not data volume. Please implement with the sample dataset provided below. Later we will plug in larger datasets — for now this is sufficient.

Goal:
 • Expand the keyword dictionary into multiple categories (confidence, hesitation, escalation, repair).
 • Apply score deltas (+2, -2, etc., configurable via YAML).
 • Wrap it as a module /dictionary.py so it can be plugged into the engine scoring.

Sample Data (use this to build/test):

[
 {"text": "I think maybe", "category": "hesitation"},
 {"text": "I'm sure of it", "category": "confidence"},
 {"text": "This is urgent!", "category": "escalation"},
 {"text": "I’m sorry about that", "category": "repair"},
 {"text": "I’m not certain", "category": "hesitation"},
 {"text": "Yes, definitely", "category": "confidence"}
]

Expected Output:
 • YAML or JSON config with categories + score weights.
 • Python module /dictionary.py that:
 • Loads config.
 • Scores an input string against categories.
 • Returns {category, score_delta}.

Reminder: This is a structural trial, not a data sufficiency trial. Please implement directly with the dataset provided.


