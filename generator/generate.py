import os
import sys
import json
import random
from concurrent.futures import ThreadPoolExecutor
import threading
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any

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

# Import shared models and setup function
from utils.common import ImageList, Story, setup_stories_db as setup_db

generation_lock = threading.Lock()
generation_count = 0
generation_limit = None
is_dry_run = False

def generate_random_image_concept(reels_data):
    if len(reels_data) < 5:
        raise ValueError("reelsData.json must contain at least 5 reels")
    
    # Randomly choose 5 distinct reels
    selected_reels = random.sample(reels_data, 5)
    
    # Randomly choose 1 image from each selected reel
    selected_images = []
    for reel in selected_reels:
        img = random.choice(reel["images"])
        selected_images.append(img)
        
    return selected_images

def process_model_group(
    model_key: str,
    group_configs: list,
    concepts: list,
    modifications: list,
    session_factory: Any,
    db_lock: threading.Lock,
    min_p_list: list = None
):
    """Processes a group of model configurations sequentially against all pre-sampled concepts"""
    global generation_count, is_dry_run
    min_p_values = min_p_list if min_p_list else [None]
    
    # Create thread-local session
    with session_factory() as db:
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

                # Run all concepts sequentially for this model configuration
                for concept in concepts:
                    concept_id = concept["id"]
                    tags = concept["tags"]
                    language = concept["language"]
                    localized_names = concept["localized_names"]
                    image_base64 = concept["image_base64"]

                    # Build variations to execute (base prompt + each modification)
                    variations = [{"id": None, "text": None}]
                    for mod in modifications:
                        mod_text = mod.get(language) or mod.get("en") or mod.get("es")
                        variations.append({"id": mod["id"], "text": mod_text})

                    # Run each prompt variation sequentially
                    for var in variations:
                        mod_id = var["id"]
                        mod_text = var["text"]
                        
                        effective_mod_id = f"{mod_id or ''}_min_p_{min_p}" if min_p is not None else mod_id
                        
                        # Run each model type (chat/vision) sequentially
                        for m_type in model_types:
                            if m_type not in ["chat", "vision"]:
                                continue
                                
                            exec_type = "vision" if m_type in ["vision"] else "chat"
                            final_model_name = f"{model_name} ({exec_type})"

                            # Check for existing story (idempotency)
                            if is_dry_run:
                                existing_story = None
                            else:
                                with db_lock:
                                    existing_story = db.query(Story).filter(
                                        Story.image_list_id == concept_id,
                                        Story.model_origin == final_model_name,
                                        Story.prompt_modification == effective_mod_id,
                                        Story.language == language
                                    ).first()
                                
                            if existing_story:
                                print(f"[{model_name}] ({exec_type}) -> Story already exists for concept {tags} with modification '{effective_mod_id or 'Base'}'. Skipping.")
                                continue

                            if generation_limit is not None:
                                with generation_lock:
                                    if generation_count >= generation_limit:
                                        return
                                    generation_count += 1
                            
                            if effective_mod_id:
                                print(f"[{model_name}] ({exec_type}) -> Applying prompt modification: '{effective_mod_id}' for concept {tags}")
                            else:
                                print(f"[{model_name}] ({exec_type}) -> Using base prompt for concept {tags}")
                            
                            try:
                                story_text, actual_final_name, metrics = common.generate_story(
                                    model_config=local_config,
                                    model_type=exec_type,
                                    tags=tags,
                                    language=language,
                                    localized_names=localized_names,
                                    image_data=image_base64 if exec_type == "vision" else None,
                                    prompt_modification_text=mod_text,
                                    script_dir=script_dir
                                )
                                
                                if not story_text:
                                    print(f"   Error: [{model_name}] ({exec_type}) generated empty story. Skipping save.")
                                    continue
                                    
                                # Save to DB (synchronized via db_lock to prevent SQLite locks)
                                think_str = common.determine_think_string(local_config, model_name)

                                if is_dry_run:
                                    print(f"   [DRY RUN] Generated story for concept {tags} successfully. Skipping database save.")
                                else:
                                    with db_lock:
                                        story_record = Story(
                                            image_list_id=concept_id,
                                            text=story_text,
                                            model_origin=final_model_name,
                                            language=language,
                                            generation_type=exec_type,
                                            execution_time=metrics.get("execution_time"),
                                            input_tokens=metrics.get("input_tokens"),
                                            output_tokens=metrics.get("output_tokens"),
                                            prompt_modification=effective_mod_id,
                                            think=think_str
                                        )
                                        db.add(story_record)
                                        db.commit()
                                        db.refresh(story_record)
                                        print(f"   Saved story to database: ID={story_record.id}, model={final_model_name}")
                                    
                            except Exception as e:
                                print(f"   Exception during story generation with model '{model_name}' ({exec_type}): {e}")
                                with db_lock:
                                    db.rollback()

def main():
    parser = argparse.ArgumentParser(description="Standalone Batch Story Generator")
    parser.add_argument("--models-config", "-m", type=str, default=str(script_dir / "model.json"), help="Path to models config JSON")
    parser.add_argument("--db-path", "-d", type=str, default=str(script_dir / "custom_stories.db"), help="Path to output SQLite database")
    parser.add_argument("--language", "-l", type=str, default=None, choices=["en", "fr", "de", "es"], help="Force stories in a specific language (en, fr, de, es)")
    parser.add_argument("--threads", "-t", type=int, default=4, help="Maximum number of parallel model execution threads")
    parser.add_argument("--dry-run", action="store_true", help="Run the generator calling the LLMs but without saving results to the database")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of story generations performed in this execution")
    parser.add_argument("--min-p", type=float, nargs="+", default=None, help="Iterate generation over list of min_p values")
    args = parser.parse_args()

    global is_dry_run, generation_limit
    is_dry_run = args.dry_run
    generation_limit = args.limit

    # Paths resolution
    models_config_path = Path(args.models_config).resolve()
    db_path = Path(args.db_path).resolve()
    reels_path = script_dir / "reelsData.json"
    icons_dir = script_dir / "icons"

    print(f"Loading reels from {reels_path}...")
    reels_data = common.load_reels_data(reels_path)

    print(f"Loading models config from {models_config_path}...")
    models_config = common.load_models_config(models_config_path)

    print(f"Setting up database at {db_path}...")
    session_factory = setup_db(db_path)
    db_lock = threading.Lock()

    # Load prompt modifications if file exists
    modifications_path = script_dir / "prompts" / "story_generation" / "modifications.json"
    modifications = []
    if modifications_path.exists():
        try:
            with open(modifications_path, 'r', encoding='utf-8') as f:
                modifications = json.load(f)
            print(f"Loaded {len(modifications)} prompt modification(s).")
        except Exception as e:
            print(f"Warning: Failed to load modifications.json: {e}")

    # Build expected model origins for configured models
    configured_origins = set()
    for model_config in models_config:
        model_name = model_config.get("name", model_config["model"])
        model_types = model_config.get("model_type", ["chat"])
        if isinstance(model_types, str):
            model_types = [model_types]
        for m_type in model_types:
            if m_type in ["chat", "vision"]:
                exec_type = "vision" if m_type in ["vision"] else "chat"
                configured_origins.add(f"{model_name} ({exec_type})")

    concepts = []
    print("Loading existing image lists from the database...")
    with session_factory() as db:
        all_img_lists = db.query(ImageList).all()
        print(f"Found {len(all_img_lists)} total image lists in the database.")
        
        for img_list in all_img_lists:
            try:
                tags = json.loads(img_list.tags)
            except Exception as e:
                continue
            
            # Determine language
            language = "es" # fallback
            existing_story = db.query(Story).filter(Story.image_list_id == img_list.id).first()
            if existing_story:
                language = existing_story.language
            elif img_list.name and "(" in img_list.name:
                for lang in ["en", "es", "fr", "de"]:
                    if f"({lang})" in img_list.name:
                        language = lang
                        break
            
            # Resolve localized names from reels
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
            localized_names = common.get_localized_names(selected_images, language)
            
            # Pre-compile the reel strip image if missing but required by any vision model
            image_base64 = img_list.image_data
            has_vision_model = any("vision" in m.get("model_type", []) or "multimodal" in m.get("model_type", []) for m in models_config)
            if has_vision_model and not image_base64:
                image_base64 = common.create_reel_strip(selected_images, icons_dir)
                if image_base64 and not is_dry_run:
                    img_list.image_data = image_base64
                    db.commit()
                    
            concepts.append({
                "id": img_list.id,
                "tags": tags,
                "language": language,
                "localized_names": localized_names,
                "image_base64": image_base64
            })
    print(f"Loaded {len(concepts)} concept(s) from the database.")

    # Dry-Run Statistics Reporting
    min_p_list = args.min_p if args.min_p else [None]
    if args.dry_run:
        print("\nCalculating statistics of stories to generate...")
        total_existing = 0
        total_missing = 0
        missing_by_model = defaultdict(int)
        existing_by_model = defaultdict(int)
        missing_details = []

        with session_factory() as db:
            for concept in concepts:
                concept_id = concept["id"]
                tags = concept["tags"]
                language = concept["language"]
                
                # Build variations
                variations = [{"id": None, "text": None}]
                for mod in modifications:
                    mod_text = mod.get(language) or mod.get("en") or mod.get("es")
                    variations.append({"id": mod["id"], "text": mod_text})
                    
                for min_p in min_p_list:
                    for model_config in models_config:
                        model_name = model_config.get("name", model_config["model"])
                        model_types = model_config.get("model_type", ["chat"])
                        if isinstance(model_types, str):
                            model_types = [model_types]
                        model_types = list(set(t.lower() for t in model_types))
                        model_types.sort()
                        
                        for m_type in model_types:
                            if m_type not in ["chat", "vision"]:
                                continue
                            exec_type = "vision" if m_type in ["vision"] else "chat"
                            final_model_name = f"{model_name} ({exec_type})"
                            
                            for var in variations:
                                mod_id = var["id"]
                                effective_mod_id = f"{mod_id or ''}_min_p_{min_p}" if min_p is not None else mod_id
                                
                                # Query DB
                                existing_story = db.query(Story).filter(
                                    Story.image_list_id == concept_id,
                                    Story.model_origin == final_model_name,
                                    Story.prompt_modification == effective_mod_id,
                                    Story.language == language
                                ).first()
                                
                                key = (final_model_name, effective_mod_id or "Base")
                                if existing_story:
                                    total_existing += 1
                                    existing_by_model[key] += 1
                                else:
                                    total_missing += 1
                                    missing_by_model[key] += 1
                                    missing_details.append({
                                        "concept_id": concept_id,
                                        "tags": tags,
                                        "language": language,
                                        "model": final_model_name,
                                        "modification": effective_mod_id or "Base"
                                    })

        print("\n" + "="*60)
        print(" STORY GENERATION BALANCE STATISTICS REPORT (DRY-RUN)")
        print("="*60)
        print(f"Total Concepts Evaluated: {len(concepts)}")
        print(f"Total Existing Stories:   {total_existing}")
        print(f"Total Missing Stories:    {total_missing}")
        print("-"*60)
        
        if total_missing == 0:
            print("Everything is balanced! No stories need to be generated.")
        else:
            print("Missing Stories by Model and Prompt Variation:")
            for key, count in sorted(missing_by_model.items()):
                model_name, mod = key
                existing = existing_by_model.get(key, 0)
                total = existing + count
                pct = (existing / total * 100) if total > 0 else 0
                print(f"  - {model_name} [{mod}]: {count} missing (Existing: {existing}/{total}, {pct:.1f}%)")
                
            print("-"*60)
            print("Sample of Missing Stories (first 10):")
            for i, detail in enumerate(missing_details[:10]):
                print(f"  {i+1}. Concept: {detail['tags']} ({detail['language']})")
                print(f"     Model:   {detail['model']}")
                print(f"     Prompt:  {detail['modification']}")
            if len(missing_details) > 10:
                print(f"  ... and {len(missing_details) - 10} more missing stories.")
        print("="*60 + "\n")

    if not concepts:
        print("No concepts to generate. Exiting.")
        return

    # Group models by their model name to run distinct models in parallel
    # while keeping different configurations/reasoning settings for the same model sequential
    grouped_models = defaultdict(list)
    for model_config in models_config:
        model_key = model_config["model"]
        grouped_models[model_key].append(model_config)

    # Start concurrent tasks using ThreadPoolExecutor
    print(f"Launching generation tasks concurrently (max {args.threads} parallel models)...")
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        for model_key, configs in grouped_models.items():
            futures.append(executor.submit(
                process_model_group,
                model_key=model_key,
                group_configs=configs,
                concepts=concepts,
                modifications=modifications,
                session_factory=session_factory,
                db_lock=db_lock,
                min_p_list=min_p_list
            ))
        
        # Wait for all thread pool tasks to finish
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error in model group task execution: {e}")

    print("\nGeneration completed successfully!")
    
    # Flush Langfuse traces if configured
    lc.flush_langfuse()

if __name__ == "__main__":
    main()
