import os
import sys
import json
import argparse
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import uuid
import re
from datetime import datetime
from pathlib import Path
from PIL import Image
import io
import base64
from collections import defaultdict
from typing import Any, List, TypedDict
from concurrent.futures import ThreadPoolExecutor
import threading

# Add this directory to sys.path to allow importing from utils package
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

import utils.common as common

# Load environment variables
common.load_env_file(script_dir)

# Import langchain utils from the local utils folder
try:
    import utils.langchain as lc
except ImportError as e:
    print(f"Failed to import utils.langchain: {e}")
    print("Ensure you run the script from the batch_generator folder or it is on the PYTHONPATH.")
    sys.exit(1)

# Import database models and setups
from utils.common import ImageList, Story, ModelNameCount, setup_stories_db, setup_reps_db

# ------------------ LANGGRAPH DEFINITION ------------------
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    print("Error: 'langgraph' is not installed. Please run: pip install -r requirements.txt")
    sys.exit(1)

class AgentState(TypedDict):
    tags: List[str]
    language: str
    localized_names: List[str]
    model_config: dict
    model_type: str
    image_data: str
    story: str
    extracted_names: List[str]
    rejected_names: List[str]
    feedback: str
    attempts: int
    success: bool
    execution_time: float
    input_tokens: int
    output_tokens: int
    messages: List[Any]
    prompt_modification_text: str

# Module-level configuration variables populated in main()
reps_session_factory = None
stories_session_factory = None
name_threshold = 3
db_lock = threading.Lock()
global_reviewer_config = None
global_total_stories = 0
global_ratio_threshold = 0.01
generation_lock = threading.Lock()
generation_count = 0
generation_limit = None
is_dry_run = False

def write_story_node(state: AgentState, config=None) -> dict:
    model_config = state["model_config"]
    model_type = state["model_type"]
    tags = state["tags"]
    language = state["language"]
    localized_names = state["localized_names"]
    image_data = state["image_data"]
    attempts = state["attempts"]
    messages = list(state.get("messages") or [])
    feedback = state.get("feedback") or ""
    
    model = common.load_model_instance(model_config)
    if not model:
        raise RuntimeError("Failed to load Writer model instance.")
        
    supports_system = model_config.get("supports_system_instructions", True)
    
    import time
    start_time = time.time()
    
    print(f"   [MultiAgent - Writer] Generating story draft (attempt {attempts + 1})...")
    
    if attempts > 0 and messages:
        # Subsequent attempt: append reviewer feedback and invoke using message history
        messages.append(HumanMessage(content=feedback))
        
        # Call generate_response using the message history as the prompt
        story_text, metrics = lc.generate_response(
            prompt=messages,
            system_prompt="",
            model=model,
            supports_system_instructions=supports_system,
            config=config
        )
        
        # Append the assistant's new response to the conversation history
        messages.append(AIMessage(content=story_text or ""))
    else:
        # First attempt: construct initial prompt and run initial LLM call
        prompt_kind = "vision" if model_type == "vision" else "chat"
        writer_prompt_path = script_dir / "prompts" / "story_generation_multiagent" / "writer" / f"{prompt_kind}_{language}.txt"
        if not writer_prompt_path.exists():
            writer_prompt_path = script_dir / "prompts" / "story_generation_multiagent" / "writer" / f"{prompt_kind}_es.txt"
            
        writer_template = common.load_prompt_content(writer_prompt_path)
        if not writer_template:
            raise RuntimeError(f"Failed to load Writer prompt template: {writer_prompt_path}")
            
        image_names_text = ", ".join(localized_names if localized_names else tags)
        prompt = writer_template.format(image_names=image_names_text)
        
        prompt_modification_text = state.get("prompt_modification_text")
        if prompt_modification_text:
            prompt = f"{prompt}\n{prompt_modification_text}"
        
        system_prompt_path = script_dir / "prompts" / "system" / f"story_generation_{language}.txt"
        if not system_prompt_path.exists():
            system_prompt_path = script_dir / "prompts" / "system" / "story_generation_es.txt"
        system_prompt = common.load_prompt_content(system_prompt_path) or "Eres un escritor creativo de historias."
        
        if image_data and model_type == "vision":
            story_text, metrics = lc.generate_multimodal_response(
                prompt_text=prompt,
                system_prompt=system_prompt,
                model=model,
                image_data=image_data,
                mime_type="image/png",
                supports_system_instructions=supports_system,
                config=config
            )
            
            # Construct initial messages list for multimodal
            if not supports_system:
                human_content = [
                    {"type": "text", "text": f"{system_prompt}\n\n{prompt}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    }
                ]
                messages = [HumanMessage(content=human_content)]
            else:
                human_content = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    }
                ]
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_content)
                ]
        else:
            story_text, metrics = lc.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                supports_system_instructions=supports_system,
                config=config
            )
            
            # Construct initial messages list for text-only
            if not supports_system:
                messages = [HumanMessage(content=f"{system_prompt}\n\n{prompt}")]
            else:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt)
                ]
                
        messages.append(AIMessage(content=story_text or ""))
        
    execution_time = (metrics.get("execution_time") or (time.time() - start_time))
    input_tokens = metrics.get("input_tokens") or 0
    output_tokens = metrics.get("output_tokens") or 0
    
    return {
        "story": story_text or "",
        "messages": messages,
        "execution_time": state["execution_time"] + execution_time,
        "input_tokens": state["input_tokens"] + input_tokens,
        "output_tokens": state["output_tokens"] + output_tokens
    }

def review_story_node(state: AgentState, config=None) -> dict:
    story = state["story"]
    language = state["language"]
    model_config = state["model_config"]
    attempts = state["attempts"]
    
    if not story:
        return {"success": False, "attempts": attempts + 1, "feedback": "Story is empty."}
        
    # Resolve reviewer prompt path
    reviewer_prompt_path = script_dir / "prompts" / "story_generation_multiagent" / "reviewer" / f"extract_{language}.txt"
    if not reviewer_prompt_path.exists():
        reviewer_prompt_path = script_dir / "prompts" / "story_generation_multiagent" / "reviewer" / f"extract_es.txt"
        
    reviewer_template = common.load_prompt_content(reviewer_prompt_path)
    if not reviewer_template:
        raise RuntimeError(f"Failed to load Reviewer prompt template: {reviewer_prompt_path}")
        
    prompt = reviewer_template.format(story=story)
    
    global global_reviewer_config
    reviewer_model_config = global_reviewer_config if global_reviewer_config else model_config
    
    model = common.load_model_instance(reviewer_model_config)
    if not model:
        raise RuntimeError("Failed to load Reviewer model instance.")
        
    from utils.langchain_parsers import ThinkStrippingJSONParser
    parser = ThinkStrippingJSONParser()
    
    import time
    start_time = time.time()
    
    print(f"   [MultiAgent - Reviewer] Extracting character names...")
    extracted_names = []
    execution_time = 0.0
    input_tokens = 0
    output_tokens = 0
    
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.messages import HumanMessage, SystemMessage
        
        sys_msg = "Extract proper names as JSON. Return valid JSON only, matching schema: {'character_names': [...]}"
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=sys_msg),
            HumanMessage(content=prompt)
        ])
        
        response_obj = model.invoke(prompt_template.format_messages(), config=config)
        
        from utils.langchain import _extract_tokens
        input_tokens, output_tokens = _extract_tokens(response_obj)
        execution_time = time.time() - start_time
        
        content = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
        
        # Parse content using parser
        parsed = parser.parse(content)
        if isinstance(parsed, dict) and "character_names" in parsed:
            extracted_names = parsed["character_names"]
        elif isinstance(parsed, list):
            extracted_names = parsed
            
    except Exception as e:
        print(f"   [MultiAgent - Reviewer] Warning: Failed to parse names JSON: {e}")
        extracted_names = []
        
    # Clean up and normalize extracted names
    normalized_names = []
    for name in extracted_names:
        clean_name = common.extract_first_name(name)
        if clean_name and len(clean_name) > 1:
            normalized_names.append(clean_name)
    normalized_names = list(set(normalized_names))
    
    print(f"   [MultiAgent - Reviewer] Extracted names (normalized): {normalized_names}")
    
    # Recount total stories dynamically from the database if session factory is available
    global reps_session_factory, stories_session_factory, global_total_stories
    if stories_session_factory:
        with db_lock:
            try:
                from sqlalchemy import text
                with stories_session_factory() as stories_db:
                    global_total_stories = stories_db.execute(text("SELECT COUNT(*) FROM stories")).scalar() or 0
            except Exception as count_err:
                print(f"   [MultiAgent - Reviewer] Warning: Failed to dynamically recount stories: {count_err}")

    # Check against forbidden list, repetitions database, and ratio threshold
    rejected_names = []
    model_name = model_config.get("name", model_config.get("model"))

    total_stories = max(100, global_total_stories)  # Use a floor of 100 to prevent volatile ratios at the start

    for name in normalized_names:
        # Check against database reps counts and global ratio
        # Use direct SQL for consistency with the atomic writes above
        with reps_session_factory() as reps_db:
            from sqlalchemy import text as sql_text
            from sqlalchemy import func

            # 1. Query model-specific repetitions
            db_cnt = reps_db.execute(
                sql_text(
                    "SELECT count FROM model_name_counts "
                    "WHERE model_origin = :model AND character_name = :name"
                ),
                {"model": model_name, "name": name}
            ).scalar() or 0

            # 2. Query GLOBAL baseline counts (pre-populated)
            global_cnt = reps_db.execute(
                sql_text(
                    "SELECT count FROM model_name_counts "
                    "WHERE model_origin = 'GLOBAL' AND character_name = :name"
                ),
                {"name": name}
            ).scalar() or 0

            # 3. Sum of repetitions across all models (excluding GLOBAL)
            db_reps_sum = reps_db.execute(
                sql_text(
                    "SELECT COALESCE(SUM(count), 0) FROM model_name_counts "
                    "WHERE character_name = :name AND model_origin != 'GLOBAL'"
                ),
                {"name": name}
            ).scalar() or 0

            # Check 1: local raw repetition count for current model
            if db_cnt >= name_threshold:
                rejected_names.append(name)
                continue

            # Check 2: true global ratio across all stories (global baseline + all models reps)
            total_cnt = global_cnt + db_reps_sum
            if (total_cnt / total_stories) >= global_ratio_threshold:
                rejected_names.append(name)
                continue
                    
    success = len(rejected_names) == 0
    feedback = ""
    
    if not success:
        rejected_str = ", ".join(rejected_names)
        feedback_path = script_dir / "prompts" / "story_generation_multiagent" / "reviewer" / f"feedback_{language}.txt"
        if not feedback_path.exists():
            feedback_path = script_dir / "prompts" / "story_generation_multiagent" / "reviewer" / "feedback_es.txt"
        feedback_template = common.load_prompt_content(feedback_path)
        if feedback_template:
            feedback = feedback_template.format(rejected_names=rejected_str)
        else:
            feedback = f"Story rejected due to overused or forbidden names: {rejected_str}. Please rewrite the story using completely different character names."
            
        print(f"   [MultiAgent - Reviewer] Story REJECTED due to names: {rejected_names}")
    else:
        print(f"   [MultiAgent - Reviewer] Story APPROVED!")
        
    return {
        "extracted_names": normalized_names,
        "rejected_names": rejected_names,
        "feedback": feedback,
        "success": success,
        "attempts": attempts + (0 if success else 1),
        "execution_time": state["execution_time"] + execution_time,
        "input_tokens": state["input_tokens"] + (input_tokens or 0),
        "output_tokens": state["output_tokens"] + (output_tokens or 0)
    }

def run_agent_workflow(
    model_config: dict,
    model_type: str,
    tags: list,
    language: str,
    localized_names: list,
    image_data: str = None,
    prompt_modification_text: str = None
):
    initial_state = {
        "tags": tags,
        "language": language,
        "localized_names": localized_names,
        "model_config": model_config,
        "model_type": model_type,
        "image_data": image_data,
        "story": "",
        "extracted_names": [],
        "rejected_names": [],
        "feedback": "",
        "attempts": 0,
        "success": False,
        "execution_time": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "messages": [],
        "prompt_modification_text": prompt_modification_text
    }
    
    workflow = StateGraph(AgentState)
    workflow.add_node("writer", write_story_node)
    workflow.add_node("reviewer", review_story_node)
    
    workflow.set_entry_point("writer")
    workflow.add_edge("writer", "reviewer")
    
    def decide_next_step(state: AgentState):
        if state["success"]:
            return "end"
        if state["attempts"] >= 3:
            return "end"
        return "writer"
        
    workflow.add_conditional_edges(
        "reviewer",
        decide_next_step,
        {
            "writer": "writer",
            "end": END
        }
    )
    
    app = workflow.compile()
    
    # Retrieve the CallbackHandler dynamically for tracing the workflow run
    handler = lc._get_langfuse_handler()
    config = {}
    if handler:
        model_name = model_config.get("name", model_config.get("model", "Model"))
        config = {
            "callbacks": [handler],
            "run_name": f"MultiAgent-Story-{model_name}-{language}"
        }
        
    final_state = app.invoke(initial_state, config=config)
    return final_state

# ------------------ EXECUTION LOOP ------------------
def process_model_group(
    model_key: str,
    group_configs: list,
    image_lists: list,
    reels_data: list,
    icons_dir: Path,
    stories_session_factory: Any,
    db_lock: threading.Lock,
    min_p_list: list = None,
    modifications: list = None
):
    global generation_count, is_dry_run
    min_p_values = min_p_list if min_p_list else [None]
    modifications_list = modifications if modifications else []
    
    with stories_session_factory() as stories_db:
        for model_config in group_configs:
            for min_p in min_p_values:
                local_config = dict(model_config)
                local_config["min_p"] = min_p
                
                model_name = local_config.get("name", local_config.get("model"))
                model_types = local_config.get("model_type", ["chat"])
                
                if isinstance(model_types, str):
                    model_types = [model_types]
                    
                model_types = list(set(t.lower() for t in model_types))
                model_types.sort()

                # Process all image lists sequentially for this model config
                for img_list in image_lists:
                    img_list_id = img_list.id
                    
                    # Parse tags
                    try:
                        tags = json.loads(img_list.tags)
                    except Exception as e:
                        print(f"Error parsing tags for image list {img_list_id}: {e}")
                        continue

                    # Get language and check database stories for this image list
                    language = "es" # default fallback
                    with db_lock:
                        existing_story = stories_db.query(Story).filter(Story.image_list_id == img_list_id).first()
                        if existing_story:
                            language = existing_story.language

                    # Resolve localized names
                    localized_names = common.get_localized_names_from_tags(tags, reels_data, language)

                    # Resolve image strip data
                    image_base64 = img_list.image_data
                    if not image_base64:
                        has_vision_model = any("vision" in m.get("model_type", []) or "multimodal" in m.get("model_type", []) for m in group_configs)
                        if has_vision_model:
                            selected_images = []
                            for tag in tags:
                                found = False
                                for reel in reels_data:
                                    for img in reel["images"]:
                                        if img["tag"] == tag:
                                            selected_images.append(img)
                                            found = True
                                            break
                                    if found:
                                        break
                            image_base64 = common.create_reel_strip(selected_images, icons_dir)
                            if image_base64 and not is_dry_run:
                                with db_lock:
                                    local_img_list = stories_db.query(ImageList).filter(ImageList.id == img_list_id).first()
                                    if local_img_list:
                                        local_img_list.image_data = image_base64
                                        stories_db.commit()

                    # Build variations (base multiagent + modifications)
                    variations = [{"id": "multiagent", "base_id": None, "text": None}]
                    for mod in modifications_list:
                        mod_text = mod.get(language) or mod.get("en") or mod.get("es")
                        variations.append({
                            "id": f"multiagent_{mod['id']}",
                            "base_id": mod["id"],
                            "text": mod_text
                        })

                    for var in variations:
                        var_id = var["id"]
                        base_id = var["base_id"]
                        mod_text = var["text"]
                        
                        base_mod_id = f"{base_id or ''}_min_p_{min_p}" if min_p is not None else base_id
                        multiagent_mod_id = f"{var_id}_min_p_{min_p}" if min_p is not None else var_id

                        # Process each model type (chat/vision) sequentially
                        for m_type in model_types:
                            if m_type not in ["chat", "vision", "multimodal"]:
                                continue
                                
                            exec_type = "vision" if m_type in ["vision", "multimodal"] else "chat"
                            final_model_name = f"{model_name} ({exec_type})"
                            # Check for existing multiagent story (idempotency)
                            if is_dry_run:
                                existing_story_multiagent = None
                            else:
                                with db_lock:
                                    existing_story_multiagent = stories_db.query(Story).filter(
                                        Story.image_list_id == img_list_id,
                                        Story.model_origin == final_model_name,
                                        Story.language == language,
                                        Story.prompt_modification == multiagent_mod_id
                                    ).first()
                            
                            if existing_story_multiagent:
                                print(f"[{model_name}] ({exec_type}) -> Story already exists with prompt_modification='{multiagent_mod_id}' for concept {tags}. Skipping.")
                                continue

                            if generation_limit is not None:
                                with generation_lock:
                                    if generation_count >= generation_limit:
                                        return
                                    generation_count += 1

                            print(f"[{model_name}] ({exec_type}) -> Launching multiagent workflow for concept {tags} with modification '{multiagent_mod_id}'...")
                            try:
                                final_state = run_agent_workflow(
                                    model_config=local_config,
                                    model_type=exec_type,
                                    tags=tags,
                                    language=language,
                                    localized_names=localized_names,
                                    image_data=image_base64 if exec_type == "vision" else None,
                                    prompt_modification_text=mod_text
                                )
                                
                                story_text = final_state.get("story")
                                if not story_text:
                                    print(f"   Error: [{model_name}] ({exec_type}) generated empty story. Skipping save.")
                                    continue

                                # Update name counts in reps DB for accepted names
                                if final_state.get("success"):
                                    accepted_names = final_state.get("extracted_names", [])
                                    if is_dry_run:
                                        print(f"   [DRY RUN] Skipped updating character name repetitions database for names: {accepted_names}")
                                    else:
                                        # Use atomic SQL UPDATE to avoid race conditions across instances
                                        with reps_session_factory() as reps_db:
                                            from sqlalchemy import text as sql_text
                                            for name in accepted_names:
                                                norm_name = name.strip().title()
                                                if not norm_name:
                                                    continue
                                                # Try to increment existing row atomically
                                                result = reps_db.execute(
                                                    sql_text(
                                                        "UPDATE model_name_counts SET count = count + 1 "
                                                        "WHERE model_origin = :model AND character_name = :name"
                                                    ),
                                                    {"model": model_name, "name": norm_name}
                                                )
                                                if result.rowcount == 0:
                                                    # No existing row — insert
                                                    reps_db.execute(
                                                        sql_text(
                                                            "INSERT OR IGNORE INTO model_name_counts "
                                                            "(model_origin, character_name, count) "
                                                            "VALUES (:model, :name, 1)"
                                                        ),
                                                        {"model": model_name, "name": norm_name}
                                                    )
                                                    # If another instance inserted first, increment instead
                                                    reps_db.execute(
                                                        sql_text(
                                                            "UPDATE model_name_counts SET count = count + 1 "
                                                            "WHERE model_origin = :model AND character_name = :name AND count = 0"
                                                        ),
                                                        {"model": model_name, "name": norm_name}
                                                    )
                                            reps_db.commit()
                                else:
                                    print(f"   Warning: [{model_name}] ({exec_type}) failed to satisfy name reviewer within attempts. Saving anyway.")

                                # Save the story to stories database
                                think_str = common.determine_think_string(local_config, model_name)

                                if is_dry_run:
                                    print(f"   [DRY RUN] Multiagent workflow completed successfully. Skipping database save.")
                                else:
                                    with db_lock:
                                        story_record = Story(
                                            image_list_id=img_list_id,
                                            text=story_text,
                                            model_origin=final_model_name,
                                            language=language,
                                            generation_type=exec_type,
                                            execution_time=final_state.get("execution_time"),
                                            input_tokens=final_state.get("input_tokens"),
                                            output_tokens=final_state.get("output_tokens"),
                                            prompt_modification=multiagent_mod_id,
                                            think=think_str,
                                            name_revisions=final_state.get("attempts", 0)
                                        )
                                        stories_db.add(story_record)
                                        stories_db.commit()
                                        stories_db.refresh(story_record)
                                        print(f"   Saved story to database: ID={story_record.id}, model={final_model_name}")

                            except Exception as e:
                                print(f"   Exception during story generation with model '{model_name}' ({exec_type}): {e}")
                                with db_lock:
                                    stories_db.rollback()

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Standalone Story Generator")
    parser.add_argument("--models-config", "-m", type=str, default=str(script_dir / "model.json"), help="Path to models config JSON")
    parser.add_argument("--reviewer-config", "-rc", type=str, default=str(script_dir / "reviewer_model.json"), help="Path to reviewer model config JSON")
    parser.add_argument("--db-path", "-d", type=str, default=str(script_dir / "custom_stories.db"), help="Path to output SQLite database")
    parser.add_argument("--reps-db-path", "-r", type=str, default=str(script_dir / "name_repetitions.db"), help="Path to output names repetitions database")
    parser.add_argument("--threshold", "-k", type=int, default=3, help="Threshold count of generations at which a name is rejected for a model")
    parser.add_argument("--threads", "-t", type=int, default=4, help="Maximum number of parallel model execution threads")
    parser.add_argument("--ratio-threshold", "-rt", type=float, default=0.01, help="Maximum ratio of total stories at which a name is rejected (e.g. 0.01 = 1%%)")
    parser.add_argument("--dry-run", action="store_true", help="Run the generator calling the LLMs but without saving results to the database")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of story generations performed in this execution")
    parser.add_argument("--min-p", type=float, nargs="+", default=None, help="Iterate generation over list of min_p values")
    args = parser.parse_args()

    global is_dry_run, generation_limit
    is_dry_run = args.dry_run
    generation_limit = args.limit

    # Paths resolution
    models_config_path = Path(args.models_config).resolve()
    reviewer_config_path = Path(args.reviewer_config).resolve()
    db_path = Path(args.db_path).resolve()
    reps_db_path = Path(args.reps_db_path).resolve()
    reels_path = script_dir / "reelsData.json"
    icons_dir = script_dir / "icons"

    global reps_session_factory, stories_session_factory, name_threshold, global_reviewer_config
    global global_total_stories, global_ratio_threshold
    
    # Initialize global variables
    name_threshold = args.threshold
    global_ratio_threshold = args.ratio_threshold

    # Load reviewer config
    print(f"Loading reviewer model config from {reviewer_config_path}...")
    try:
        with open(reviewer_config_path, 'r', encoding='utf-8') as f:
            global_reviewer_config = json.load(f)
        print(f"Loaded reviewer configuration successfully: model={global_reviewer_config.get('model')}")
    except Exception as e:
        print(f"Warning: Failed to load reviewer model config from {reviewer_config_path}: {e}")
        print("Will fallback to using the writer's model configuration for review.")
        global_reviewer_config = None

    print(f"Loading reels from {reels_path}...")
    reels_data = common.load_reels_data(reels_path)

    print(f"Loading models config from {models_config_path}...")
    models_config = common.load_models_config(models_config_path)

    print(f"Setting up stories database at {db_path}...")
    stories_session_factory = setup_stories_db(db_path)
    
    # Count total stories in the database
    from sqlalchemy import text
    try:
        with stories_session_factory() as db:
            global_total_stories = db.execute(text("SELECT COUNT(*) FROM stories")).scalar() or 0
        print(f"Total existing stories in database: {global_total_stories}")
    except Exception as e:
        print(f"Warning: Failed to count existing stories: {e}")
        global_total_stories = 0
    
    print(f"Setting up repetitions database at {reps_db_path}...")
    reps_session_factory = setup_reps_db(reps_db_path)

    # Load prompt modifications if file exists
    modifications_path = script_dir / "prompts" / "story_generation_multiagent" / "writer" / "modifications.json"
    modifications = []
    if modifications_path.exists():
        try:
            with open(modifications_path, 'r', encoding='utf-8') as f:
                modifications = json.load(f)
            print(f"Loaded {len(modifications)} prompt modification(s).")
        except Exception as e:
            print(f"Warning: Failed to load modifications.json: {e}")

    # Load all image lists from the database
    with stories_session_factory() as db:
        image_lists = db.query(ImageList).all()
        
    print(f"Found {len(image_lists)} image list(s) in the database.")
    
    if not image_lists:
        print("No image lists found. Please run generate.py first to sample image lists and populate the database.")
        return

    print(f"Active concepts: {len(image_lists)}")

    # Dry-Run Statistics Reporting
    min_p_list = args.min_p if args.min_p else [None]
    if args.dry_run:
        print("\nCalculating statistics of multiagent stories to generate...")
        total_existing = 0
        total_missing = 0
        missing_by_model = defaultdict(int)
        existing_by_model = defaultdict(int)
        missing_details = []

        with stories_session_factory() as db:
            for img_list in image_lists:
                try:
                    tags = json.loads(img_list.tags)
                except Exception as e:
                    continue
                
                # Determine language
                language = "es" # default fallback
                existing_story = db.query(Story).filter(Story.image_list_id == img_list.id).first()
                if existing_story:
                    language = existing_story.language
                elif img_list.name and "(" in img_list.name:
                    for lang in ["en", "es", "fr", "de"]:
                        if f"({lang})" in img_list.name:
                            language = lang
                            break

                for min_p in min_p_list:
                    # Build variations (base multiagent + modifications)
                    variations = [{"id": "multiagent", "base_id": None}]
                    for mod in modifications:
                        variations.append({
                            "id": f"multiagent_{mod['id']}",
                            "base_id": mod["id"]
                        })

                    for var in variations:
                        var_id = var["id"]
                        base_id = var["base_id"]
                        
                        base_mod_id = f"{base_id or ''}_min_p_{min_p}" if min_p is not None else base_id
                        multiagent_mod_id = f"{var_id}_min_p_{min_p}" if min_p is not None else var_id
                        
                        for model_config in models_config:
                            model_name = model_config.get("name", model_config["model"])
                            model_types = model_config.get("model_type", ["chat"])
                            if isinstance(model_types, str):
                                model_types = [model_types]
                            model_types = list(set(t.lower() for t in model_types))
                            model_types.sort()
                            
                            for m_type in model_types:
                                if m_type not in ["chat", "vision", "multimodal"]:
                                    continue
                                exec_type = "vision" if m_type in ["vision", "multimodal"] else "chat"
                                final_model_name = f"{model_name} ({exec_type})"
                                
                                # Query DB for multiagent story
                                existing_story = db.query(Story).filter(
                                    Story.image_list_id == img_list.id,
                                    Story.model_origin == final_model_name,
                                    Story.language == language,
                                    Story.prompt_modification == multiagent_mod_id
                                ).first()
                                
                                key = f"{final_model_name} (min_p={min_p})"
                                if existing_story:
                                    total_existing += 1
                                    existing_by_model[key] += 1
                                else:
                                    total_missing += 1
                                    missing_by_model[key] += 1
                                    missing_details.append({
                                        "tags": tags,
                                        "language": language,
                                        "model": key
                                    })

        print("\n" + "="*60)
        print(" MULTIAGENT STORY GENERATION BALANCE STATISTICS REPORT (DRY-RUN)")
        print("="*60)
        print(f"Total Concepts Evaluated: {len(image_lists)}")
        print(f"Total Existing Stories:   {total_existing}")
        print(f"Total Missing Stories:    {total_missing}")
        print("-"*60)
        
        if total_missing == 0:
            print("Everything is balanced! No multiagent stories need to be generated.")
        else:
            print("Missing Multiagent Stories by Model:")
            for model_name, count in sorted(missing_by_model.items()):
                existing = existing_by_model.get(model_name, 0)
                total = existing + count
                pct = (existing / total * 100) if total > 0 else 0
                print(f"  - {model_name}: {count} missing (Existing: {existing}/{total}, {pct:.1f}%)")
                
            print("-"*60)
            print("Sample of Missing Stories (first 10):")
            for i, detail in enumerate(missing_details[:10]):
                print(f"  {i+1}. Concept: {detail['tags']} ({detail['language']})")
                print(f"     Model:   {detail['model']}")
            if len(missing_details) > 10:
                print(f"  ... and {len(missing_details) - 10} more missing stories.")
        print("="*60 + "\n")

    # Group models by their model name to run distinct models in parallel
    grouped_models = defaultdict(list)
    for model_config in models_config:
        model_key = model_config["model"]
        grouped_models[model_key].append(model_config)

    # Start concurrent tasks using ThreadPoolExecutor (max workers from args.threads)
    print(f"Launching multi-agent generation tasks concurrently (max {args.threads} parallel models)...")
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        for model_key, configs in grouped_models.items():
            futures.append(executor.submit(
                process_model_group,
                model_key=model_key,
                group_configs=configs,
                image_lists=image_lists,
                reels_data=reels_data,
                icons_dir=icons_dir,
                stories_session_factory=stories_session_factory,
                db_lock=db_lock,
                min_p_list=min_p_list,
                modifications=modifications
            ))
        
        # Wait for all thread pool tasks to finish
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error in model group task execution: {e}")

    print("\nMulti-Agent generation completed successfully!")
    
    # Flush Langfuse traces if configured
    lc.flush_langfuse()

if __name__ == "__main__":
    main()
