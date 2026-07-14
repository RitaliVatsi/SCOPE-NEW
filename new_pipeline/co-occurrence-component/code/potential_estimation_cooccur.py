"""
Co-occurrence-aware potential estimation (Gap 2 in ../../GAPS.md).

Drop-in variant of src/potential_estimation_gpt_goal.py. Same request/retry
machinery, but the prompt sent to GPT-4o gains two things the original never
had:

1. Chain context - which landmarks have already been confirmed earlier in
   this run (from chain_object_prompt.py's mental_map), so the model knows
   this frontier is being scored as part of a sequence, not in isolation.
2. A ROOM_PLAUSIBILITY field - a semantic object/room co-occurrence prior
   ("dishwashers co-occur with kitchens/sinks; a mirror is unlikely near a
   dining table"), elicited zero-shot from GPT-4o's own commonsense, not a
   trained co-occurrence model. This is the DIV-Nav-inspired prior discussed
   in GAPS.md Gap 2.

Deliberately NOT a spatial-proximity prior - GAPS.md's empirical run showed
chains jump between rooms (mirror in a bathroom, everything else elsewhere),
so "boost near the last landmark's position" would be actively wrong.

This module is standalone: it does not modify src/potential_estimation_gpt_goal.py,
so run_object_goal.py and run_goatbench_evaluation.py are unaffected. Only
chain_object_prompt_cooccur.py imports this.
"""

import openai
from openai import OpenAI
from src.const import *
from typing import Optional, List, Dict
from PIL import Image
import io
import base64
import time
import os

client = OpenAI(
    base_url=END_POINT,
    api_key=OPENAI_KEY,
)


def _format_chain_context(chain_context: Optional[List[Dict]]) -> str:
    """Render already-confirmed landmarks as an ordered route summary."""
    if not chain_context:
        return "**Route so far**: this is the first landmark in the chain; nothing confirmed yet."

    ordered = sorted(chain_context, key=lambda e: e.get("found_step") or 0)
    names = ", ".join(e["name"] for e in ordered)
    return (
        f"**Route so far**: already confirmed, in order: {names}. "
        "Treat this as the sequence of rooms/areas already visited on this run."
    )


def format_content(image, question_text, question_image_path, chain_context=None):
    img_pil = Image.fromarray(image.astype('uint8'))
    with io.BytesIO() as output:
        img_pil.save(output, format="PNG")
        png_bytes = output.getvalue()
    frontier_image = base64.b64encode(png_bytes).decode("utf-8")

    # Read and encode the question image
    question_image = None
    if question_image_path is not None and os.path.exists(question_image_path):
        try:
            question_img_pil = Image.open(question_image_path)
            with io.BytesIO() as output:
                question_img_pil.save(output, format="PNG")
                question_png_bytes = output.getvalue()
            question_image = base64.b64encode(question_png_bytes).decode("utf-8")
        except Exception as e:
            print(f"Error loading question image: {e}")

    formated_content = []
    formated_content.append({"type": "text", "text": (
        "You are a semantic reasoning agent assisting a robot in navigation planning. "
        "The robot is considering the following **frontier observation** as a potential place to visit. "
        "You are also given the **goal question**, the **route so far** (landmarks already confirmed "
        "earlier in this same run, if any), and, if available, the **goal image**. Your task is to analyze "
        "whether exploring this frontier would help the robot achieve its goal.\n\n"
        "Analyze the frontier image and provide ratings for each criterion. Use EXACTLY this format:\n\n"
        "**SEMANTIC_RICHNESS:** [Low/Medium/High]\n"
        "**EXPLORABILITY:** [Low/Medium/High]\n"
        "**GOAL_RELEVANCE:** [Low/Medium/High]\n"
        "**ROOM_PLAUSIBILITY:** [Low/Medium/High]\n"
        "**POTENTIAL_SCORE:** [X.X] (where X.X is a number from 1.0 to 5.0)\n"
        "**EXPLANATION:** [Your reasoning in 2-3 sentences]\n\n"
        "Definitions:\n"
        "- Semantic richness: How many meaningful objects, structures, or environmental cues are visible?\n"
        "- Explorability: Does this lead to new regions, paths, or unexplored areas? Are there doors, corridors, stairs?\n"
        "- Goal relevance: Based on the goal, does this frontier likely contain or lead to the target?\n"
        "- Room plausibility: Using ONLY commonsense semantic co-occurrence (not the route so far's spatial "
        "location - landmark chains commonly jump between unrelated rooms), how plausible is it that the room "
        "type visible or implied in this frontier is one that would typically contain the current goal? "
        "E.g. a dishwasher plausibly co-occurs with kitchens/sinks, not bedrooms; a mirror plausibly co-occurs "
        "with bathrooms/bedrooms, not a garage. Rate Low if the visible room type is a poor semantic fit for "
        "the goal, High if it's a strong typical fit, Medium if ambiguous or the room type isn't identifiable.\n"
        "- Potential score: Overall value for exploration (1.0=very low, 3.0=medium, 5.0=very high)\n\n"
        "Example output:\n"
        "**SEMANTIC_RICHNESS:** High\n"
        "**EXPLORABILITY:** Medium\n"
        "**GOAL_RELEVANCE:** High\n"
        "**ROOM_PLAUSIBILITY:** High\n"
        "**POTENTIAL_SCORE:** 4.2\n"
        "**EXPLANATION:** The image shows a hallway leading to a kitchen-like area with counters and a sink, "
        "which commonly co-occurs with the goal object. This is semantically rich and highly relevant."
    )})

    formated_content.append({"type": "text", "text": f"**Goal question**: {question_text}"})
    formated_content.append({"type": "text", "text": _format_chain_context(chain_context)})

    if question_image is not None:
        formated_content.append({"type": "text", "text": "**Goal image**:"})
        formated_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{question_image}",
                    "detail": "high",
                },
            }
        )
    else:
        formated_content.append({"type": "text", "text": "**Goal image**: Not provided"})

    formated_content.append({"type": "text", "text": "**Frontier observation**:"})

    formated_content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{frontier_image}",
                "detail": "high",
            },
        }
    )
    return formated_content


# send information to openai
def get_potential_estimation(metadata, image) -> Optional[str]:
    rate_limit_retries = 0
    other_error_retries = 0
    max_rate_limit_retries = 30  # More generous for potential estimation
    max_other_error_retries = 15  # More retries for this critical function

    question_text = metadata['question']
    question_image_path = metadata['image'] if 'image' in metadata else None
    chain_context = metadata.get('chain_context')
    formated_content = format_content(image, question_text, question_image_path, chain_context)
    message_text = [
        {"role": "user", "content": formated_content},
    ]

    while True:  # Keep trying indefinitely for rate limits
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=message_text,
                temperature=0.7,
                max_tokens=4096,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            rate_limit_retries += 1
            wait_time = min(60 + (rate_limit_retries * 10), 300)  # Exponential backoff, max 5 minutes
            print(f"Rate limit error in potential estimation ({rate_limit_retries}), waiting {wait_time}s before retry...")
            time.sleep(wait_time)

            # If we've hit too many rate limits, give a longer break
            if rate_limit_retries >= max_rate_limit_retries:
                print(f"Hit {max_rate_limit_retries} rate limits, taking a 15-minute break...")
                time.sleep(900)  # 15 minute break
                rate_limit_retries = 0  # Reset counter after long break
            continue
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
            other_error_retries += 1
            if other_error_retries > max_other_error_retries:
                print(f"Too many connection/timeout/server errors in potential estimation ({other_error_retries}), using default scores")
                return None  # Fallback to default scores in potential graph
            wait_time = min(30 + (other_error_retries * 15), 180)  # Exponential backoff, max 3 minutes
            print(f"API connection/timeout/server error in potential estimation ({other_error_retries}), waiting {wait_time}s before retry: {e}")
            time.sleep(wait_time)
            continue
        except openai.BadRequestError as e:
            print(f"Bad request error in potential estimation (likely permanent): {e}")
            return None
        except Exception as e:
            other_error_retries += 1
            if other_error_retries > max_other_error_retries:
                print(f"Too many unexpected errors in potential estimation ({other_error_retries}), using default scores: {e}")
                return None
            wait_time = min(30 + (other_error_retries * 15), 180)
            print(f"Unexpected error in potential estimation ({other_error_retries}), waiting {wait_time}s before retry: {e}")
            time.sleep(wait_time)
            continue
