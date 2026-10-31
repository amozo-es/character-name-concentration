import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from PIL import Image
import io
import base64
from collections import defaultdict
from typing import Any
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

# Import database models and setup helper
from utils.common import ImageList, Story, setup_stories_db as setup_db

generation_lock = threading.Lock()
generation_count = 0
generation_limit = None
is_dry_run = False

def process_model_group(
    model_key: str,
    group_configs: list,
    image_lists: list,
    reels_data: list,
    icons_dir: Path,
    session_factory: Any,
    db_lock: threading.Lock,
    min_p_list: list = None
):
    global generation_count, is_dry_run
    min_p_values = min_p_list if min_p_list else [None]
    
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
                        existing_story = db.query(Story).filter(Story.image_list_id == img_list_id).first()
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
                                    local_img_list = db.query(ImageList).filter(ImageList.id == img_list_id).first()
                                    if local_img_list:
                                        local_img_list.image_data = image_base64
                                        db.commit()

                    # Process each model type (chat/vision) sequentially
                    for m_type in model_types:
                        if m_type not in ["chat", "vision", "multimodal"]:
                            continue
                            
                        exec_type = "vision" if m_type in ["vision", "multimodal"] else "chat"
                        final_model_name = f"{model_name} ({exec_type})"
                        
                        base_mod_id = f"_min_p_{min_p}" if min_p is not None else None
                        multiphase_mod_id = f"multiphase_min_p_{min_p}" if min_p is not None else "multiphase"
                        
                        # Check for existing multiphase story (idempotency)
                        if is_dry_run:
                            existing_story_multiphase = None
                        else:
                            with db_lock:
                                existing_story_multiphase = db.query(Story).filter(
                                    Story.image_list_id == img_list_id,
                                    Story.model_origin == final_model_name,
                                    Story.language == language,
                                    Story.prompt_modification == multiphase_mod_id
                                ).first()
                        
                        if existing_story_multiphase:
                            print(f"[{model_name}] ({exec_type}) -> Story already exists with prompt_modification='{multiphase_mod_id}' for concept {tags}. Skipping.")
                            continue

                        if generation_limit is not None:
                            with generation_lock:
                                if generation_count >= generation_limit:
                                    return
                                generation_count += 1

                        print(f"[{model_name}] ({exec_type}) -> Launching multiphase generation for concept {tags}...")
                        try:
                            story_text, actual_final_name, metrics = common.generate_multiphase_story(
                                model_config=local_config,
                                model_type=exec_type,
                                tags=tags,
                                language=language,
                                localized_names=localized_names,
                                image_data=image_base64 if exec_type == "vision" else None,
                                script_dir=script_dir
                            )
                            
                            if not story_text:
                                print(f"   Error: [{model_name}] ({exec_type}) generated empty story. Skipping save.")
                                continue

                            # Determine think string
                            think_str = common.determine_think_string(local_config, model_name)

                            if is_dry_run:
                                print(f"   [DRY RUN] Generated multiphase story successfully. Skipping database save.")
                            else:
                                with db_lock:
                                    story_record = Story(
                                        image_list_id=img_list_id,
                                        text=story_text,
                                        model_origin=actual_final_name,
                                        language=language,
                                        generation_type=exec_type,
                                        execution_time=metrics.get("execution_time"),
                                        input_tokens=metrics.get("input_tokens"),
                                        output_tokens=metrics.get("output_tokens"),
                                        prompt_modification=multiphase_mod_id,
                                        think=think_str
                                    )
                                    db.add(story_record)
                                    db.commit()
                                    db.refresh(story_record)
                                    print(f"   Saved story to database: ID={story_record.id}, model={actual_final_name}")

                        except Exception as e:
                            print(f"   Exception during story generation with model '{model_name}' ({exec_type}): {e}")
                            with db_lock:
                                db.rollback()

def main():
    parser = argparse.ArgumentParser(description="Multi-Phase Standalone Story Generator")
    parser.add_argument("--models-config", "-m", type=str, default=str(script_dir / "model.json"), help="Path to models config JSON")
    parser.add_argument("--db-path", "-d", type=str, default=str(script_dir / "custom_stories.db"), help="Path to output SQLite database")
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

    # Load all image lists from the database
    with session_factory() as db:
        image_lists = db.query(ImageList).all()
        
    print(f"Found {len(image_lists)} total image list(s) in the database.")
    print(f"Active concepts: {len(image_lists)}")

    if not image_lists:
        print("No image lists found/selected. Exiting.")
        return

    # Dry-Run Statistics Reporting
    min_p_list = args.min_p if args.min_p else [None]
    if args.dry_run:
        print("\nCalculating statistics of multiphase stories to generate...")
        total_existing = 0
        total_missing = 0
        missing_by_model = defaultdict(int)
        existing_by_model = defaultdict(int)
        missing_details = []

        with session_factory() as db:
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
                    base_mod_id = f"_min_p_{min_p}" if min_p is not None else None
                    multiphase_mod_id = f"multiphase_min_p_{min_p}" if min_p is not None else "multiphase"
                    
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
                            
                            # Query DB for multiphase story
                            existing_story = db.query(Story).filter(
                                Story.image_list_id == img_list.id,
                                Story.model_origin == final_model_name,
                                Story.language == language,
                                Story.prompt_modification == multiphase_mod_id
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
        print(" MULTIPHASE STORY GENERATION BALANCE STATISTICS REPORT (DRY-RUN)")
        print("="*60)
        print(f"Total Concepts Evaluated: {len(image_lists)}")
        print(f"Total Existing Stories:   {total_existing}")
        print(f"Total Missing Stories:    {total_missing}")
        print("-"*60)
        
        if total_missing == 0:
            print("Everything is balanced! No multiphase stories need to be generated.")
        else:
            print("Missing Multiphase Stories by Model:")
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
    print(f"Launching multi-phase generation tasks concurrently (max {args.threads} parallel models)...")
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

    print("\nMulti-Phase generation completed successfully!")
    
    # Flush Langfuse traces if configured
    lc.flush_langfuse()

if __name__ == "__main__":
    main()
