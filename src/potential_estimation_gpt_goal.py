import openai
from openai import OpenAI
from src.const import *
from typing import Optional
from PIL import Image
import io
import base64
import time
import os

client = OpenAI(
    base_url=END_POINT,
    api_key=OPENAI_KEY,
)


def format_content(image, question_text, question_image_path):
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
        "The robot is considering the following **frontier observation** as a potential place to visit. You are also given the **goal question** and, if available, the **goal image**. Your task is to analyze whether exploring this frontier would help the robot achieve its goal.\n\n"
        "Analyze the frontier image and provide ratings for each criterion. Use EXACTLY this format:\n\n"
        "**SEMANTIC_RICHNESS:** [Low/Medium/High]\n"
        "**EXPLORABILITY:** [Low/Medium/High]\n" 
        "**GOAL_RELEVANCE:** [Low/Medium/High]\n"
        "**POTENTIAL_SCORE:** [X.X] (where X.X is a number from 1.0 to 5.0)\n"
        "**EXPLANATION:** [Your reasoning in 2-3 sentences]\n\n"
        "Definitions:\n"
        "- Semantic richness: How many meaningful objects, structures, or environmental cues are visible?\n"
        "- Explorability: Does this lead to new regions, paths, or unexplored areas? Are there doors, corridors, stairs?\n"
        "- Goal relevance: Based on the goal, does this frontier likely contain or lead to the target?\n"
        "- Potential score: Overall value for exploration (1.0=very low, 3.0=medium, 5.0=very high)\n\n"
        "Example output:\n"
        "**SEMANTIC_RICHNESS:** High\n"
        "**EXPLORABILITY:** Medium\n"
        "**GOAL_RELEVANCE:** High\n"
        "**POTENTIAL_SCORE:** 4.2\n"
        "**EXPLANATION:** The image shows a hallway with several doorways leading to rooms with furniture. This is semantically rich and highly relevant for finding household objects."
    )})

    formated_content.append({"type": "text", "text": f"**Goal question**: {question_text}"})

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
    formated_content = format_content(image, question_text, question_image_path)
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