import os
import sys
import json
import argparse
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Any
from concurrent.futures import ThreadPoolExecutor

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

# Import shared models and setup helpers
from utils.common import ImageList, Story, setup_stories_db as setup_db

generation_lock = threading.Lock()
generation_count = 0
generation_limit = None
is_dry_run = False

NO_STIMULUS_UUID = "00000000-0000-0000-0000-000000000000"

def ensure_dummy_image_list(session_factory):
    """Ensures that the special UUID record exists in the image_lists table"""
    with session_factory() as db:
        dummy = db.query(ImageList).filter(ImageList.id == NO_STIMULUS_UUID).first()
        if not dummy:
            print(f"Creating dummy ImageList for stories without stimulus (ID: {NO_STIMULUS_UUID})...")
            dummy = ImageList(
                id=NO_STIMULUS_UUID,
                tags="[]",
                name="Without Stimulus (No Images)",
                image_data=None,
                created_at=datetime.utcnow()
            )
            db.add(dummy)
            db.commit()

def process_model_group(
    model_key: str,
    group_configs: list,
    modifications: list,
    session_factory: Any,
    db_lock: threading.Lock,
    min_p_list: list = None
):
    """Processes a group of model configurations sequentially to generate the remaining stories"""
    global generation_count, is_dry_run
    min_p_values = min_p_list if min_p_list else [None]
    
    with session_factory() as db:
        for model_config in group_configs:
            for min_p in min_p_values:
                local_config = dict(model_config)
                local_config["min_p"] = min_p
                
                model_name = local_config.get("name", local_config.get("model"))
                final_model_name = f"{model_name} (chat)"

                # Define languages and targets
                languages_targets = {
                    "en": 40,
                    "es": 20,
                    "fr": 20,
                    "de": 20
                }

                # Build variations
                variations = [{"id": "without_stimulus", "text": None}]
                for mod in modifications:
                    # We will check variations if any modifications are present
                    # However, for stories without stimulus, we focus primarily on the base prompt
                    # But support variations to remain consistent with generate.py
                    for lang in languages_targets.keys():
                        mod_text = mod.get(lang) or mod.get("en") or mod.get("es")
                        variations.append({"id": mod["id"], "text": mod_text})

                for var in variations:
                    mod_id = var["id"]
                    mod_text = var["text"]
                    
                    effective_mod_id = f"{mod_id or ''}_min_p_{min_p}" if min_p is not None else mod_id

                    for language, target in languages_targets.items():
                        # Count existing stories for this combination in DB
                        if is_dry_run:
                            existing_count = 0
                        else:
                            with db_lock:
                                existing_count = db.query(Story).filter(
                                    Story.image_list_id == NO_STIMULUS_UUID,
                                    Story.model_origin == final_model_name,
                                    Story.prompt_modification == effective_mod_id,
                                    Story.language == language
                                ).count()

                        needed = target - existing_count
                        if needed <= 0:
                            continue

                        print(f"[{model_name}] (chat) -> Modification: '{effective_mod_id or 'Base'}'. Language: {language.upper()}. Target: {target}, Existing: {existing_count}. Generating {needed} story(ies)...")

                        for i in range(needed):
                            if generation_limit is not None:
                                with generation_lock:
                                    if generation_count >= generation_limit:
                                        return
                                    generation_count += 1
                            try:
                                print(f"   Generating story {existing_count + i + 1}/{target}...")
                                story_text, actual_final_name, metrics = common.generate_story(
                                    model_config=local_config,
                                    language=language,
                                    prompt_modification_text=mod_text,
                                    without_stimulus=True,
                                    script_dir=script_dir
                                )
                                
                                if not story_text:
                                    print(f"   Error: generated empty story. Skipping save.")
                                    continue

                                # Save to DB
                                think_str = common.determine_think_string(local_config, model_name)

                                if is_dry_run:
                                    print(f"   [DRY RUN] Generated story successfully. Skipping database save.")
                                else:
                                    with db_lock:
                                        story_record = Story(
                                            image_list_id=NO_STIMULUS_UUID,
                                            text=story_text,
                                            model_origin=final_model_name,
                                            language=language,
                                            generation_type="chat",
                                            execution_time=metrics.get("execution_time"),
                                            input_tokens=metrics.get("input_tokens"),
                                            output_tokens=metrics.get("output_tokens"),
                                            prompt_modification=effective_mod_id,
                                            think=think_str
                                        )
                                        db.add(story_record)
                                        db.commit()
                                        db.refresh(story_record)
                                        print(f"   Saved story ID={story_record.id}")

                            except Exception as e:
                                print(f"   Exception during story generation: {e}")
                                with db_lock:
                                    db.rollback()

def main():
    parser = argparse.ArgumentParser(description="Batch Story Generator Without Stimulus")
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

    models_config_path = Path(args.models_config).resolve()
    db_path = Path(args.db_path).resolve()

    print(f"Loading models config from {models_config_path}...")
    models_config = common.load_models_config(models_config_path)

    print(f"Setting up database at {db_path}...")
    session_factory = setup_db(db_path)
    db_lock = threading.Lock()

    print("Ensuring dummy image list record exists...")
    ensure_dummy_image_list(session_factory)

    # Load prompt modifications if file exists
    modifications_path = script_dir / "prompts" / "story_without_stimulus" / "modifications.json"
    modifications = []
    if modifications_path.exists():
        try:
            with open(modifications_path, 'r', encoding='utf-8') as f:
                modifications = json.load(f)
            print(f"Loaded {len(modifications)} prompt modification(s).")
        except Exception as e:
            print(f"Warning: Failed to load modifications.json: {e}")

    languages_targets = {
        "en": 40,
        "es": 20,
        "fr": 20,
        "de": 20
    }

    # Statistics reporting
    min_p_list = args.min_p if args.min_p else [None]
    if args.dry_run:
        print("\n" + "="*60)
        print(" DRY-RUN: STORY GENERATION WITHOUT STIMULUS BALANCE REPORT")
        print("="*60)
        
        with session_factory() as db:
            for model_config in models_config:
                model_name = model_config.get("name", model_config["model"])
                final_model_name = f"{model_name} (chat)"
                
                # Build variations
                variations = [{"id": "without_stimulus", "text": None}]
                for mod in modifications:
                    variations.append({"id": mod["id"], "text": "exists"})

                print(f"\nModel: {final_model_name}")
                for min_p in min_p_list:
                    print(f"  Min_P: {min_p}")
                    for var in variations:
                        mod_id = var["id"]
                        effective_mod_id = f"{mod_id or ''}_min_p_{min_p}" if min_p is not None else mod_id
                        print(f"    Variation: '{effective_mod_id or 'Base'}'")
                        for language, target in languages_targets.items():
                            existing_count = db.query(Story).filter(
                                Story.image_list_id == NO_STIMULUS_UUID,
                                Story.model_origin == final_model_name,
                                Story.prompt_modification == effective_mod_id,
                                Story.language == language
                            ).count()
                            needed = max(0, target - existing_count)
                            pct = (existing_count / target) * 100
                            print(f"      - {language.upper()}: {existing_count}/{target} stories ({pct:.1f}%) | Needed: {needed}")
        print("="*60 + "\n")

    # Group models by model name
    grouped_models = defaultdict(list)
    for model_config in models_config:
        model_key = model_config["model"]
        grouped_models[model_key].append(model_config)

    # Start ThreadPoolExecutor
    print(f"Launching generation tasks concurrently (max {args.threads} parallel models)...")
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        for model_key, configs in grouped_models.items():
            futures.append(executor.submit(
                process_model_group,
                model_key=model_key,
                group_configs=configs,
                modifications=modifications,
                session_factory=session_factory,
                db_lock=db_lock,
                min_p_list=min_p_list
            ))
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error in model group task execution: {e}")

    print("\nGeneration of stories without stimulus completed successfully!")
    
    # Flush Langfuse traces if configured
    lc.flush_langfuse()

if __name__ == "__main__":
    main()
