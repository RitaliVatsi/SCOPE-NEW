import openai
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import os
import time
from typing import Optional
import logging
from src.const import *

client = OpenAI(
    base_url=END_POINT,
    api_key=OPENAI_KEY,
)

def format_content(contents):
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{c[1]}",
                        "detail": "high",
                    },
                }
            )
    return formated_content


# send information to openai
def call_openai_api(sys_prompt, contents) -> Optional[str]:
    rate_limit_retries = 0
    other_error_retries = 0
    max_rate_limit_retries = 20
    max_other_error_retries = 10
    
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]
    
    while True:  # Keep trying indefinitely for rate limits
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-2024-11-20",  # model = "deployment_name"
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
            wait_time = min(60 + (rate_limit_retries * 10), 300)
            print(f"Rate limit error ({rate_limit_retries}), waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            
            if rate_limit_retries >= max_rate_limit_retries:
                print(f"Hit {max_rate_limit_retries} rate limits, taking a 10-minute break...")
                time.sleep(600)
                rate_limit_retries = 0
            continue
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
            other_error_retries += 1
            if other_error_retries > max_other_error_retries:
                print(f"Too many connection/timeout/server errors ({other_error_retries}), giving up")
                return None
            wait_time = min(30 + (other_error_retries * 15), 180)
            print(f"API connection/timeout/server error ({other_error_retries}), waiting {wait_time}s before retry: {e}")
            time.sleep(wait_time)
            continue
        except openai.BadRequestError as e:
            print(f"Bad request error (likely permanent): {e}")
            return None
        except Exception as e:
            other_error_retries += 1
            if other_error_retries > max_other_error_retries:
                print(f"Too many unexpected errors ({other_error_retries}), giving up: {e}")
                return None
            wait_time = min(30 + (other_error_retries * 15), 180)
            print(f"Unexpected error ({other_error_retries}), waiting {wait_time}s before retry: {e}")
            time.sleep(wait_time)
            continue


# encode tensor images to base64 format
def encode_tensor2base64(img):
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


def format_question(step):
    question = step["question"]
    image_goal = None
    if "task_type" in step and step["task_type"] == "image":
        with open(step["image"], "rb") as image_file:
            image_goal = base64.b64encode(image_file.read()).decode("utf-8")

    return question, image_goal


def get_step_info(step, verbose=False):
    # 1 get question data
    question, image_goal = format_question(step)

    # 2 get step information(egocentric, frontier, snapshot)
    # 2.1 get egocentric views
    egocentric_imgs = []
    if step.get("use_egocentric_views", False):
        for egocentric_view in step["egocentric_views"]:
            egocentric_imgs.append(encode_tensor2base64(egocentric_view))

    # 2.2 get frontiers
    frontier_imgs = []
    for frontier in step["frontier_imgs"]:
        frontier_imgs.append(encode_tensor2base64(frontier))

    # Get potential scores for frontiers
    frontier_potential_scores = step.get("frontier_potential_scores", [])

    # 2.3 get snapshots
    snapshot_classes = {}  # rgb_id -> list of classes
    snapshot_full_imgs = {}  # rgb_id -> full img
    snapshot_crops = {}  # rgb_id -> list of crops
    snapshot_clusters = {}  # rgb_id -> list of clusters
    obj_map = step["obj_map"]
    seen_classes = set()
    for i, rgb_id in enumerate(step["snapshot_imgs"].keys()):
        snapshot_img = step["snapshot_imgs"][rgb_id]["full_img"]
        snapshot_full_imgs[rgb_id] = encode_tensor2base64(snapshot_img)
        snapshot_crops[rgb_id] = [
            encode_tensor2base64(crop_data["crop"])
            for crop_data in step["snapshot_imgs"][rgb_id]["object_crop"]
        ]
        snapshot_class = [
            crop_data["obj_class"]
            for crop_data in step["snapshot_imgs"][rgb_id]["object_crop"]
        ]
        cluster_class = [
            obj_map[int(obj_id)] for obj_id in step["snapshot_objects"][rgb_id]
        ]
        # remove duplicates
        seen_classes.update(sorted(list(set(snapshot_class))))
        snapshot_classes[rgb_id] = snapshot_class
        snapshot_clusters[rgb_id] = cluster_class

    # 3 prefiltering, note that we need the obj_id_mapping
    keep_index = list(range(len(snapshot_full_imgs)))
    keep_index_snapshot = {
        rgb_id: list(range(len(snapshot_crops[rgb_id]))) for rgb_id in snapshot_crops
    }
    if step.get("use_prefiltering") is True:
        use_full_obj_list = step["use_full_obj_list"]
        n_prev_snapshot = len(snapshot_full_imgs)
        snapshot_classes, keep_index, keep_index_snapshot = prefiltering(
            question,
            snapshot_classes,
            snapshot_clusters,
            seen_classes,
            step["top_k_categories"],
            image_goal,
            use_full_obj_list,
            verbose=verbose,
        )
        snapshot_full_imgs = {
            rgb_id: snapshot_full_imgs[rgb_id] for rgb_id in keep_index_snapshot.keys()
        }
        for rgb_id in snapshot_classes.keys():
            snapshot_crops[rgb_id] = [
                snapshot_crops[rgb_id][i] for i in keep_index_snapshot[rgb_id]
            ]
        if verbose:
            logging.info(
                f"Prefiltering snapshot: {n_prev_snapshot} -> {len(snapshot_full_imgs)}"
            )

    return (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        keep_index,
        keep_index_snapshot,
        frontier_potential_scores,
    )


def format_explore_prompt(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    snapshot_crops,
    frontier_potential_scores=None,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
):
    sys_prompt = "Task: You are an agent in an indoor scene that is able to observe the surroundings and explore the environment. You are tasked with indoor navigation, and you are required to choose either a Snapshot or a Frontier image to explore and find the target object required in the question.\n"

    content = []
    # 1 here is some basic info
    text = "Definitions:\n"
    text += (
        "Snapshot: A focused observation of several objects. It contains a full image of the cluster of objects, and separate image crops of each object. "
        + "Choosing a snapshot means that the object asked in the question is within the cluster of objects that the snapshot represents, and you will choose that object as the final answer of the question. "
        + "Therefore, if you choose a snapshot, you should also choose the object in the snapshot that you think is the answer to the question.\n"
    )
    text += "Frontier: An unexplored region that could potentially lead to new information for answering the question. Selecting a frontier means that you will further explore that direction.\n"

    if frontier_potential_scores and any(score > 0 for score in frontier_potential_scores):
        text += "Potential Score: Each frontier has an associated potential score (0.0-5.0) indicating its estimated value for exploration based on semantic richness, explorability, and goal relevance. Higher scores suggest more promising exploration directions.\n"

    # 2 here is the question
    text += f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))

    text = "Select the Frontier/Snapshot that would help find the answer of the question.\n"
    content.append((text,))

    # 3 here is the egocentric views
    if egocentric_view:
        text = (
            "The following is the egocentric view of the agent in forward direction: "
        )
        content.append((text, egocentric_imgs[-1]))
        content.append(("\n",))

    # 4 here is the snapshot images
    text = "The followings are all the snapshots that you can choose. Following each snapshot image are the class name and image crop of each object contained in the snapshot.\n"
    text += "Please note that the class name may not be accurate due to the limitation of the object detection model. "
    text += "So you still need to utilize the images to make the decision.\n"
    content.append((text,))
    if len(snapshot_imgs) == 0:
        content.append(("No Snapshot is available\n",))
    else:
        for i, rgb_id in enumerate(snapshot_imgs.keys()):
            content.append((f"Snapshot {i} ", snapshot_imgs[rgb_id]))
            for j in range(len(snapshot_crops[rgb_id])):
                content.append(
                    (
                        f"Object {j}: {snapshot_classes[rgb_id][j]}",
                        snapshot_crops[rgb_id][j],
                    )
                )
            content.append(("\n",))

    # 5 here is the frontier images
    text = "The followings are all the Frontiers that you can explore: \n"
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available\n",))
    else:
        for i in range(len(frontier_imgs)):
            if frontier_potential_scores and i < len(frontier_potential_scores):
                potential_score = frontier_potential_scores[i]
                content.append((f"Frontier {i} (Potential Score: {potential_score:.2f}) ", frontier_imgs[i]))
            else:
                content.append((f"Frontier {i} ", frontier_imgs[i]))
            content.append(("\n",))

    # 6 here is the format of the answer
    text = "Please provide your answer in the following format: 'Snapshot i, Object j' or 'Frontier i', where i, j are the index of the snapshot or frontier you choose. "
    text += "For example, if you choose the fridge in the first snapshot, please return 'Snapshot 0, Object 2', where 2 is the index of the fridge in that snapshot.\n"
    text += "You can explain the reason for your choice, but put it in a new line after the choice.\n"
    content.append((text,))

    return sys_prompt, content


def format_prefiltering_prompt(question, class_list, top_k=10, image_goal=None):
    content = []
    sys_prompt = "You are an AI agent in a 3D indoor scene.\n"
    prompt = "Your goal is to answer questions about the scene through exploration.\n"
    prompt += "To efficiently solve the problem, you should first rank objects in the scene based on their importance.\n"
    prompt += "These are the rules for the task.\n"
    prompt += "1. Read through the whole object list.\n"
    prompt += "2. Rank objects in the list based on how well they can help your exploration given the question.\n"
    prompt += f"3. Reprint the name of all objects that may help your exploration given the question. "
    prompt += "4. Do not print any object not included in the list or include any additional information in your response.\n"
    content.append((prompt,))
    # ------------------format an example-------------------------
    prompt = "Here is an example of selecting helpful objects:\n"
    prompt += "Question: What can I use to watch my favorite shows and movies?\n"
    prompt += (
        "Following is a list of objects that you can choose, each object one line\n"
    )
    prompt += "painting\nspeaker\nbox\ncabinet\nlamp\ntv\nbook rack\nsofa\noven\nbed\ncurtain\n"
    prompt += "Answer: tv\nspeaker\nsofa\nbed\n"
    content.append((prompt,))
    # ------------------Task to solve----------------------------
    prompt = f"Following is the concrete content of the task and you should retrieve helpful objects in order:\n"
    prompt += f"Question: {question}"
    if image_goal is not None:
        content.append((prompt, image_goal))
        content.append(("\n",))
    else:
        content.append((prompt + "\n",))
    prompt = (
        "Following is a list of objects that you can choose, each object one line\n"
    )
    for i, cls in enumerate(class_list):
        prompt += f"{cls}\n"
    prompt += "Answer: "
    content.append((prompt,))
    return sys_prompt, content


def get_prefiltering_classes(question, seen_classes, top_k=10, image_goal=None):
    prefiltering_sys, prefiltering_content = format_prefiltering_prompt(
        question, sorted(list(seen_classes)), top_k=top_k, image_goal=image_goal
    )

    message = ""
    for c in prefiltering_content:
        message += c[0]
        if len(c) == 2:
            message += f": image {c[1][:10]}..."
    
    response = call_openai_api(prefiltering_sys, prefiltering_content)
    if response is None:
        print("Prefiltering API failed completely, using all classes")
        return sorted(list(seen_classes))[:top_k]

    selected_classes = response.strip().split("\n")
    selected_classes = [cls.strip() for cls in selected_classes]
    selected_classes = [cls for cls in selected_classes if cls in seen_classes]
    selected_classes = selected_classes[:top_k]

    return selected_classes


def prefiltering(
    question,
    snapshot_classes,
    snapshot_clusters,
    seen_classes,
    top_k=10,
    image_goal=None,
    use_full_obj_list=False,
    verbose=False,
):
    selected_classes = get_prefiltering_classes(
        question, seen_classes, top_k, image_goal
    )
    if verbose:
        logging.info(f"Prefiltering selected classes: {selected_classes}")

    keep_index = [
        i
        for i, k in enumerate(snapshot_clusters.keys())
        if len(set(snapshot_clusters[k]) & set(selected_classes)) > 0
    ]
    keep_snapshot_id = [list(snapshot_classes.keys())[i] for i in keep_index]
    snapshot_classes = {rgb_id: snapshot_classes[rgb_id] for rgb_id in keep_snapshot_id}

    keep_index_snapshot = {}
    for rgb_id in keep_snapshot_id:
        keep_index_snapshot[rgb_id] = [
            i
            for i in range(len(snapshot_classes[rgb_id]))
            if snapshot_classes[rgb_id][i] in selected_classes
        ]
        snapshot_classes[rgb_id] = [
            snapshot_classes[rgb_id][i] for i in keep_index_snapshot[rgb_id]
        ]

    return snapshot_classes, keep_index, keep_index_snapshot


def format_self_refine_prompt(question, snapshot_img, description, selected_object_class, task_type="description", image_goal=None):

    sys_prompt = "You are an AI agent reviewing your previous navigation choice. Your task is to validate whether your previous selection is correct given the question and the snapshot you chose."
    
    content = []
    text = f"Question: {question}\n\n"
    
    # For image tasks, include the reference image first
    if task_type == "image" and image_goal is not None:
        text += f"Reference image that you need to match:"
        content.append((text, image_goal))
        text = f"\nYou previously selected a snapshot containing '{selected_object_class}' as your answer.\n"
        text += f"Here is the snapshot image you selected:"
        content.append((text, snapshot_img))
    else:
        text += f"You previously selected a snapshot containing '{selected_object_class}' as your answer.\n"
        text += f"Here is the snapshot image you selected:"
        content.append((text, snapshot_img))
    
    text = f"\nIMPORTANT: By selecting a snapshot (rather than a frontier), you are claiming that the target object IS PRESENT in this snapshot and that '{selected_object_class}' IS the target object you were looking for.\n"
    text += f"If the target object is not in any available snapshots, you should have chosen a frontier for further exploration instead.\n\n"
    
    if task_type == "object":
        text += f"Task Type: Object Navigation - Find a specific category of object.\n\n"
        text += "Please carefully examine the snapshot image and reconsider your choice.\n"
        text += f"Does this snapshot actually contain an object of the requested category, and is '{selected_object_class}' truly an instance of that category?\n"
        text += f"Remember: you need to find any instance of the target object category. If this object doesn't match the requested category, you should have chosen a frontier instead.\n\n"
    elif task_type == "description":
        text += f"Task Type: Description-based Navigation - Find an object matching a specific language description.\n\n"
        text += "Please carefully examine the snapshot image and reconsider your choice.\n"
        text += f"Does this snapshot actually contain an object that matches both the category and the specific detailed description given in the question?\n"
        text += f"Is '{selected_object_class}' truly the object described in the question?\n"
        text += f"Remember: the object must match the specific descriptive characteristics mentioned, not just the general category. "
        text += f"Pay attention to details like color, size, shape, material, position, or other distinguishing features mentioned in the description.\n"
        text += f"If no object in this snapshot matches the description, you should have chosen a frontier for further exploration.\n\n"
    elif task_type == "image":
        text += f"Task Type: Image-based Navigation - Find an object that matches a reference image.\n\n"
        text += "Please carefully examine both the reference image and the snapshot image, and reconsider your choice.\n"
        text += f"Does this snapshot actually contain the exact same object instance that was shown in the reference image?\n"
        text += f"Is '{selected_object_class}' truly the same object from the reference image?\n"
        text += f"Pay attention to the specific details, colors, textures, and shape of the object.\n"
        text += f"Remember: you need to find the specific object instance from the reference image, not just a similar-looking object.\n"
        text += f"If the exact object from the reference image is not in this snapshot, you should have chosen a frontier for further exploration.\n\n"
    else:
        raise Exception(f"Unsupported task type: {task_type}")
    
    text += "Respond with either:\n"
    text += "- 'CONFIRM' if this snapshot truly contains the target object and your selection is correct\n" 
    text += "- 'REJECT' if this snapshot does not contain the target object (meaning you should have chosen a frontier instead)\n\n"
    text += "After your decision (CONFIRM or REJECT), you can provide a brief explanation on a new line."
    content.append((text,))
    
    return sys_prompt, content


def self_refine_choice(question, snapshot_img, description, selected_object_class, task_type="description", image_goal=None, verbose=False):

    sys_prompt, content = format_self_refine_prompt(
        question, snapshot_img, description, selected_object_class, task_type, image_goal
    )
    
    if verbose:
        logging.info(f"Performing self-refinement validation for {task_type} task...")
    
    response = call_openai_api(sys_prompt, content)
    
    if response is None:
        if verbose:
            logging.info("Self-refine API call failed completely, defaulting to CONFIRM")
        return True
    
    response = response.strip().upper()
    
    if response.startswith('CONFIRM'):
        if verbose:
            logging.info(f"Self-refine: CONFIRMED choice for {task_type} task")
        return True
    elif response.startswith('REJECT'):
        if verbose:
            logging.info(f"Self-refine: REJECTED choice for {task_type} task")
        return False
    else:
        if verbose:
            logging.info(f"Self-refine: Unclear response '{response}', defaulting to CONFIRM")
        return True


def explore_step(step, cfg, verbose=False):
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_categories"] = cfg.top_k_categories
    (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        snapshot_id_mapping,
        snapshot_crop_mapping,
        frontier_potential_scores,
    ) = get_step_info(step, verbose)
    
    # Log input statistics
    if verbose:
        logging.info(f"Input data: {len(snapshot_full_imgs)} snapshots, {len(frontier_imgs)} frontiers")
        logging.info(f"Potential scores: {frontier_potential_scores}")
    
    sys_prompt, content = format_explore_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        frontier_potential_scores=frontier_potential_scores,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
    )

    if verbose:
        logging.info(f"Input prompt length: {len(sys_prompt)} chars")

    retry_bound = 5
    max_cycle_count = 3
    max_total_iterations = 10
    total_iterations = 0
    final_response = None
    final_reason = None
    
    # Track choice history for cycle detection
    choice_history = {}
    
    # Check if we have any valid choices available
    if len(snapshot_full_imgs) == 0 and len(frontier_imgs) == 0:
        logging.error("No snapshots or frontiers available for VLM choice")
        return None, snapshot_id_mapping, snapshot_crop_mapping, "No choices available", 0
    
    while retry_bound > 0 and total_iterations < max_total_iterations:
        retry_bound -= 1
        total_iterations += 1
        
        if verbose:
            logging.info(f"VLM API call attempt {total_iterations}, retries remaining: {retry_bound}")
        
        response = call_openai_api(sys_prompt, content)

        if response is None:
            logging.warning(f"API call failed, retries remaining: {retry_bound}")
            if retry_bound == 0:
                logging.error("All API retry attempts exhausted")
                return None, snapshot_id_mapping, snapshot_crop_mapping, "API failed", len(snapshot_full_imgs)
            continue

        if verbose:
            logging.info(f"Raw VLM response: {response[:200]}...")

        response = response.strip()
        if "\n" in response:
            response_parts = response.split("\n")
            response, reason = response_parts[0], response_parts[-1]
        else:
            reason = ""
        
        original_response = response
        response = response.lower()
        
        try:
            choice_parts = response.split(",")[0].strip().split(" ")
            if len(choice_parts) != 2:
                raise ValueError(f"Expected 2 parts, got {len(choice_parts)}: {choice_parts}")
            choice_type, choice_id = choice_parts
        except Exception as e:
            logging.warning(f"Error parsing response format '{original_response}': {e}, retries remaining: {retry_bound}")
            if retry_bound == 0:
                logging.error(f"Failed to parse response after all retries: {original_response}")
                return None, snapshot_id_mapping, snapshot_crop_mapping, f"Parse error: {original_response}", len(snapshot_full_imgs)
            continue

        response_valid = False
        object_choice_id = None
        
        # Validate choice format and bounds
        if choice_type == "snapshot":
            if not choice_id.isdigit():
                logging.warning(f"Invalid snapshot ID format: {choice_id}, retries remaining: {retry_bound}")
                if retry_bound == 0:
                    return None, snapshot_id_mapping, snapshot_crop_mapping, f"Invalid snapshot ID: {choice_id}", len(snapshot_full_imgs)
                continue
            
            choice_id_int = int(choice_id)
            if not (0 <= choice_id_int < len(snapshot_full_imgs)):
                logging.warning(f"Snapshot ID out of range: {choice_id_int} (max: {len(snapshot_full_imgs)-1}), retries remaining: {retry_bound}")
                if retry_bound == 0:
                    return None, snapshot_id_mapping, snapshot_crop_mapping, f"Snapshot ID out of range: {choice_id_int}", len(snapshot_full_imgs)
                continue
            
            # Parse object choice
            try:
                object_parts = response.split(",")[1].strip().split(" ")
                if len(object_parts) != 2:
                    raise ValueError(f"Expected 2 object parts, got {len(object_parts)}")
                object_choice_type, object_choice_id = object_parts
            except Exception as e:
                logging.warning(f"Error parsing object choice from '{original_response}': {e}, retries remaining: {retry_bound}")
                if retry_bound == 0:
                    return None, snapshot_id_mapping, snapshot_crop_mapping, f"Object parse error: {original_response}", len(snapshot_full_imgs)
                continue
            
            if object_choice_type != "object":
                logging.warning(f"Invalid object choice type: {object_choice_type}, retries remaining: {retry_bound}")
                if retry_bound == 0:
                    return None, snapshot_id_mapping, snapshot_crop_mapping, f"Invalid object type: {object_choice_type}", len(snapshot_full_imgs)
                continue
            
            if not object_choice_id.isdigit():
                logging.warning(f"Invalid object ID format: {object_choice_id}, retries remaining: {retry_bound}")
                if retry_bound == 0:
                    return None, snapshot_id_mapping, snapshot_crop_mapping, f"Invalid object ID: {object_choice_id}", len(snapshot_full_imgs)
                continue
            
            object_choice_id_int = int(object_choice_id)
            snapshot_key = list(snapshot_crop_mapping.keys())[choice_id_int]
            max_object_id = len(snapshot_crop_mapping[snapshot_key]) - 1
            
            if not (0 <= object_choice_id_int <= max_object_id):
                logging.warning(f"Object ID out of range: {object_choice_id_int} (max: {max_object_id}), retries remaining: {retry_bound}")
                if retry_bound == 0:
                    return None, snapshot_id_mapping, snapshot_crop_mapping, f"Object ID out of range: {object_choice_id_int}", len(snapshot_full_imgs)
                continue
            
            response_valid = True
            
        elif choice_type == "frontier":
            if not choice_id.isdigit():
                logging.warning(f"Invalid frontier ID format: {choice_id}, retries remaining: {retry_bound}")
                if retry_bound == 0:
                    return None, snapshot_id_mapping, snapshot_crop_mapping, f"Invalid frontier ID: {choice_id}", len(snapshot_full_imgs)
                continue
            
            choice_id_int = int(choice_id)
            if not (0 <= choice_id_int < len(frontier_imgs)):
                logging.warning(f"Frontier ID out of range: {choice_id_int} (max: {len(frontier_imgs)-1}), retries remaining: {retry_bound}")
                if retry_bound == 0:
                    return None, snapshot_id_mapping, snapshot_crop_mapping, f"Frontier ID out of range: {choice_id_int}", len(snapshot_full_imgs)
                continue
            
            response_valid = True
        else:
            logging.warning(f"Invalid choice type: {choice_type}, retries remaining: {retry_bound}")
            if retry_bound == 0:
                return None, snapshot_id_mapping, snapshot_crop_mapping, f"Invalid choice type: {choice_type}", len(snapshot_full_imgs)
            continue

        if response_valid:
            if choice_type == "snapshot":
                choice_key = (choice_type, choice_id, object_choice_id)
            else:
                choice_key = (choice_type, choice_id, None)
            
            rejection_count = choice_history.get(choice_key, 0)
            
            if (choice_type == "snapshot" and 
                hasattr(cfg, 'use_self_refine') and cfg.use_self_refine):
                
                if rejection_count >= max_cycle_count:
                    if verbose:
                        logging.info(f"Choice {choice_key} has been rejected {rejection_count} times, accepting to break cycle")
                    final_response = original_response.lower()
                    final_reason = reason
                    break
                
                snapshot_idx = int(choice_id)
                object_idx = int(object_choice_id)
                
                if snapshot_idx >= len(snapshot_full_imgs):
                    logging.error(f"Snapshot index {snapshot_idx} out of range (max: {len(snapshot_full_imgs)-1})")
                    continue
                
                snapshot_rgb_id = list(snapshot_full_imgs.keys())[snapshot_idx]
                snapshot_img_b64 = snapshot_full_imgs[snapshot_rgb_id]
                
                if object_idx >= len(snapshot_classes[snapshot_rgb_id]):
                    logging.error(f"Object index {object_idx} out of range for snapshot {snapshot_rgb_id}")
                    continue
                
                selected_object_class = snapshot_classes[snapshot_rgb_id][object_idx]
                
                task_type = step.get("task_type", "object")
                description = step.get("question", "")
                
                if verbose:
                    logging.info(f"Performing self-refinement validation for {task_type} task")
                
                choice_confirmed = self_refine_choice(
                    question, snapshot_img_b64, description, selected_object_class, task_type, image_goal, verbose
                )
                
                if choice_confirmed:
                    if verbose:
                        logging.info(f"Self-refine confirmed snapshot choice")
                    final_response = original_response.lower()
                    final_reason = reason
                    break
                else:
                    choice_history[choice_key] = rejection_count + 1
                    if verbose:
                        logging.info(f"Self-refine rejected choice (count: {choice_history[choice_key]}), retrying...")
                    
                    if choice_history[choice_key] == 1:
                        additional_instruction = f"\n\nNote: Please carefully consider your choice. If the target object is not clearly visible in any snapshot, consider choosing a frontier for further exploration."
                    elif choice_history[choice_key] == 2:
                        rejected_class = selected_object_class
                        additional_instruction = f"\n\nNote: A previous choice of '{rejected_class}' was not suitable. Either select a different object that better matches the target, or choose a frontier if the target is not visible."
                    else:
                        additional_instruction = f"\n\nNote: Please make your best choice. Consider whether the target object is truly present in the available snapshots or if frontier exploration is needed."
                    
                    modified_content = content.copy()
                    if modified_content and isinstance(modified_content[-1], tuple):
                        last_text = modified_content[-1][0]
                        modified_content[-1] = (last_text + additional_instruction,)
                    content = modified_content
                    continue
            else:
                if verbose:
                    logging.info(f"Accepting {choice_type} choice (no self-refine needed)")
                final_response = original_response.lower()
                final_reason = reason
                break

    if final_response is None:
        logging.error(f"All retry attempts exhausted after {total_iterations} iterations, no valid response obtained")
        return None, snapshot_id_mapping, snapshot_crop_mapping, "All retries exhausted", len(snapshot_full_imgs)

    if verbose:
        logging.info(f"Final response after {total_iterations} iterations: {final_response}")

    return (
        final_response,
        snapshot_id_mapping,
        snapshot_crop_mapping,
        final_reason,
        len(snapshot_full_imgs),
    )
