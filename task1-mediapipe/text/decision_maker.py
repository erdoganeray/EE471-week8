"""MediaPipe Decision Maker: answer yes/no, multiple-choice and rubric-score questions about a text."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import mediapipe as mp
from common import base_options

d = mp.tasks.decision
# the text we ask questions about (pass your own text as command-line arguments)
REVIEW = "I love this phone! The camera is stunning, it is blazing fast and the battery easily lasts all day."
if len(sys.argv) > 1:
    REVIEW = " ".join(sys.argv[1:])
print("text:", REVIEW)

# Step 1: choose the model
opts = d.DecisionMakerOptions(base_options=base_options("laya_s256.task"))
with d.DecisionMaker.create_from_options(opts) as dm:
    # Step 2: Boolean question = yes/no; returns value, probability of "true" and confidence
    b = dm.evaluate_boolean(REVIEW, d.BooleanQuestion(condition="The reviewer mentions a hardware problem."))
    print(f"\nBoolean: value={b.value} p(true)={b.probability_true:.2f} confidence={b.confidence:.2f}")

    # Step 3: Choice question = pick the best-fitting option; returns a probability per option
    q = d.ChoiceQuestion(criteria={
        "positive": "The review is mostly positive.",
        "negative": "The review is mostly negative.",
        "mixed": "The review contains both clear praise and clear complaints.",
    })
    c = dm.evaluate_choice(REVIEW, q)
    print(f"Choice : {c.selected_key} probs={ {k: round(v, 2) for k, v in c.probabilities.items()} }")

    # Step 4: Score question = rubric ordered from worst (index 0) to best (index 4)
    s = dm.evaluate_score(REVIEW, d.ScoreQuestion(rubric=[
        "The reviewer hates the product.",
        "The reviewer is disappointed with the product.",
        "The reviewer is neutral about the product.",
        "The reviewer likes the product.",
        "The reviewer loves the product.",
    ]))
    print(f"Score  : expected={s.expected_score:.2f} (0 = rubric[0] ... 4 = rubric[4]) selected={s.selected_key!r}")
