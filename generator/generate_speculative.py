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
from utils.common import ImageList, Story, SlopTracker, setup_stories_db as setup_db, setup_intermediate_db

generation_lock = threading.Lock()
generation_count = 0
generation_limit = None
is_dry_run = False

# Module-level slop fingerprints (loaded once in main)
global_slop_fingerprints = None


def process_model_group(
    model_key: str,
    group_configs: list,
    image_lists: list,
    reels_data: list,
    icons_dir: Path,
    session_factory: Any,
    db_lock: threading.Lock,
    db_path: Path,
    lambdas: list,
    hard_mode: bool,
    hard_threshold: float,
    min_p_list: list = None,
    intermediate_db_path: Path = None,
):
    global generation_count, is_dry_run, global_slop_fingerprints
    min_p_values = min_p_list if min_p_list else [None]

    # Per-cell trackers: key = (final_model_name, full_mod_id) -> SlopTracker
    trackers = {}

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
                    language = "es"  # default fallback
                    with db_lock:
                        existing_story = db.query(Story).filter(Story.image_list_id == img_list_id).first()
                        if existing_story:
                            language = existing_story.language

                    # Resolve localized names
                    localized_names = common.get_localized_names_from_tags(tags, reels_data, language)

                    # Resolve image strip data
                    image_base64 = img_list.image_data
                    if not image_base64:
                        has_vision_model = any(
                            "vision" in m.get("model_type", []) or "multimodal" in m.get("model_type", [])
                            for m in group_configs
                        )
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
                                    local_img_list = db.query(ImageList).filter(
                                        ImageList.id == img_list_id
                                    ).first()
                                    if local_img_list:
                                        local_img_list.image_data = image_base64
                                        db.commit()

                    # Process each model type (chat/vision) sequentially
                    for m_type in model_types:
                        if m_type not in ["chat", "vision", "multimodal"]:
                            continue

                        exec_type = "vision" if m_type in ["vision", "multimodal"] else "chat"
                        final_model_name = f"{model_name} ({exec_type})"

                        # Build the list of speculative variants to generate
                        variants = []
                        for lam in lambdas:
                            mod_id = f"speculative_l{int(lam):02d}"
                            variants.append((mod_id, lam, False))
                        if hard_mode:
                            variants.append(("speculative_hard", None, True))

                        for mod_id, lam_val, is_hard in variants:
                            # Append min_p suffix if applicable
                            if min_p is not None:
                                full_mod_id = f"{mod_id}_min_p_{min_p}"
                            else:
                                full_mod_id = mod_id

                            # Check for existing story (idempotency)
                            if is_dry_run:
                                existing_story_spec = None
                            else:
                                with db_lock:
                                    existing_story_spec = db.query(Story).filter(
                                        Story.image_list_id == img_list_id,
                                        Story.model_origin == final_model_name,
                                        Story.language == language,
                                        Story.prompt_modification == full_mod_id
                                    ).first()

                            if existing_story_spec:
                                print(
                                    f"[{model_name}] ({exec_type}) -> "
                                    f"Story already exists with prompt_modification='{full_mod_id}' "
                                    f"for concept {tags}. Skipping."
                                )
                                continue

                            if generation_limit is not None:
                                with generation_lock:
                                    if generation_count >= generation_limit:
                                        return
                                    generation_count += 1

                            print(
                                f"[{model_name}] ({exec_type}) -> "
                                f"Launching speculative naming (mod={full_mod_id}) for concept {tags}..."
                            )

                            # Get or create per-cell tracker
                            cell_key = (final_model_name, full_mod_id)
                            if cell_key not in trackers:
                                fp = global_slop_fingerprints.get(final_model_name, {}) if global_slop_fingerprints else {}
                                baseline_freqs = fp.get("freqs", {})
                                trackers[cell_key] = SlopTracker(
                                    model_origin=final_model_name,
                                    prompt_modification=full_mod_id,
                                    baseline_freqs=baseline_freqs,
                                    db_path=db_path,
                                )
                            tracker = trackers[cell_key]

                            try:
                                story_text, raw_response, names_match, actual_final_name, metrics = \
                                    common.generate_speculative_story(
                                        model_config=local_config,
                                        model_type=exec_type,
                                        tags=tags,
                                        language=language,
                                        localized_names=localized_names,
                                        image_data=image_base64 if exec_type == "vision" else None,
                                        script_dir=script_dir,
                                        tracker=tracker,
                                        lambda_value=lam_val if lam_val is not None else 10.0,
                                        hard_mode=is_hard,
                                        hard_threshold=hard_threshold,
                                        intermediate_db_path=intermediate_db_path,
                                        intermediate_meta={
                                            "image_list_id": img_list_id,
                                            "prompt_modification": full_mod_id,
                                        },
                                    )

                                if not story_text:
                                    print(
                                        f"   Error: [{model_name}] ({exec_type}) "
                                        f"generated empty story. Skipping save."
                                    )
                                    continue

                                # Determine think string
                                think_str = common.determine_think_string(local_config, model_name)

                                if is_dry_run:
                                    print(
                                        f"   [DRY RUN] Generated speculative story successfully. "
                                        f"Skipping database save."
                                    )
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
                                            prompt_modification=full_mod_id,
                                            raw_response=raw_response,
                                            names_match=names_match,
                                            selected_names=metrics.get("selected_names"),
                                            think=think_str,
                                        )
                                        db.add(story_record)
                                        db.commit()
                                        db.refresh(story_record)
                                        print(
                                            f"   Saved story to database: "
                                            f"ID={story_record.id}, model={actual_final_name}, "
                                            f"mod={full_mod_id}"
                                        )

                                    # Persist tracker after each story
                                    tracker.save()

                            except Exception as e:
                                print(
                                    f"   Exception during speculative generation with model "
                                    f"'{model_name}' ({exec_type}): {e}"
                                )
                                with db_lock:
                                    db.rollback()


def main():
    parser = argparse.ArgumentParser(
        description="Speculative Naming Generator — draft-verify-reject at the name level"
    )
    parser.add_argument(
        "--models-config", "-m", type=str, default=str(script_dir / "model.json"),
        help="Path to models config JSON"
    )
    parser.add_argument(
        "--db-path", "-d", type=str, default=str(script_dir / "custom_stories.db"),
        help="Path to output SQLite database"
    )
    parser.add_argument(
        "--slop-fingerprints", type=str,
        default=str(script_dir / "slop_fingerprints.json"),
        help="Path to slop_fingerprints.json (from generate_slop_fingerprints.py)"
    )
    parser.add_argument(
        "--threads", "-t", type=int, default=4,
        help="Maximum number of parallel model execution threads"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the generator calling the LLMs but without saving results to the database"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit the number of story generations performed in this execution"
    )
    parser.add_argument(
        "--min-p", type=float, nargs="+", default=None,
        help="Iterate generation over list of min_p values (e.g., 0.02)"
    )
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=[0, 5, 10, 20],
        help="Lambda values for the speculative acceptance rule (default: 0 5 10 20). "
             "0 = pure verbalized (control), higher = more aggressive anti-slop."
    )
    parser.add_argument(
        "--hard", action="store_true",
        help="Also generate the HARD variant (threshold-based greedy accept)"
    )
    parser.add_argument(
        "--hard-threshold", type=float, default=0.02,
        help="Slop frequency threshold for HARD mode (default: 0.02 = 2%%)"
    )
    parser.add_argument(
        "--intermediate-db", type=str, default=None,
        help="Path to intermediate calls DB. Default: <db_path-dir>/intermediate_calls.db"
    )
    args = parser.parse_args()

    global is_dry_run, generation_limit, global_slop_fingerprints
    is_dry_run = args.dry_run
    generation_limit = args.limit

    # Paths resolution
    models_config_path = Path(args.models_config).resolve()
    db_path = Path(args.db_path).resolve()
    slop_fp_path = Path(args.slop_fingerprints).resolve()
    reels_path = script_dir / "reelsData.json"
    icons_dir = script_dir / "icons"

    # Intermediate DB path (default: same dir as stories DB)
    if args.intermediate_db:
        intermediate_db_path = Path(args.intermediate_db).resolve()
    else:
        intermediate_db_path = db_path.parent / "intermediate_calls.db"
    setup_intermediate_db(intermediate_db_path)

    # Load slop fingerprints
    print(f"Loading slop fingerprints from {slop_fp_path}...")
    if not slop_fp_path.exists():
        print(f"Error: {slop_fp_path} does not exist.")
        print("Run generate_slop_fingerprints.py first to create it.")
        sys.exit(1)
    with open(slop_fp_path, "r", encoding="utf-8") as f:
        global_slop_fingerprints = json.load(f)
    print(f"Loaded fingerprints for {len(global_slop_fingerprints)} models.")

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

    # Print configuration summary
    min_p_list = args.min_p if args.min_p else [None]
    print("\n" + "=" * 70)
    print(" SPECULATIVE NAMING GENERATION CONFIGURATION")
    print("=" * 70)
    print(f"  Lambda values:    {args.lambdas}")
    print(f"  Hard mode:        {args.hard} (threshold={args.hard_threshold})")
    print(f"  min_p values:     {min_p_list}")
    print(f"  Models:           {len(models_config)}")
    print(f"  Image lists:      {len(image_lists)}")
    print(f"  Threads:          {args.threads}")

    # Calculate total stories to generate
    total_planned = 0
    variants_per_cell = len(args.lambdas) + (1 if args.hard else 0)
    for model_config in models_config:
        model_types = model_config.get("model_type", ["chat"])
        if isinstance(model_types, str):
            model_types = [model_types]
        model_types = list(set(t.lower() for t in model_types))
        valid_types = [t for t in model_types if t in ["chat", "vision", "multimodal"]]
        total_planned += len(valid_types) * len(image_lists) * variants_per_cell * len(min_p_list)

    # Build the exact set of prompt_modification IDs this run will generate
    # so the "already exist" count only reflects matching configurations
    base_mod_ids = [f"speculative_l{int(lam):02d}" for lam in args.lambdas]
    if args.hard:
        base_mod_ids.append("speculative_hard")
    if min_p_list and min_p_list != [None]:
        target_mod_ids = []
        for mid in base_mod_ids:
            for mp in min_p_list:
                target_mod_ids.append(f"{mid}_min_p_{mp}")
    else:
        target_mod_ids = base_mod_ids

    existing_count = 0
    with session_factory() as db:
        from sqlalchemy import func
        config_model_names = set()
        for mc in models_config:
            mc_name = mc.get("name", mc.get("model"))
            mc_types = mc.get("model_type", ["chat"])
            if isinstance(mc_types, str):
                mc_types = [mc_types]
            for mt in set(t.lower() for t in mc_types):
                if mt in ["chat", "vision", "multimodal"]:
                    exec_type = "vision" if mt in ["vision", "multimodal"] else "chat"
                    config_model_names.add(f"{mc_name} ({exec_type})")

        existing_count = db.query(func.count(Story.id)).filter(
            Story.prompt_modification.in_(target_mod_ids),
            Story.model_origin.in_(config_model_names),
        ).scalar() or 0

    print(f"  Target mods:      {target_mod_ids}")
    print(f"  Total planned:    {total_planned} stories ({existing_count} already exist for this config)")
    if args.limit:
        print(f"  Limit:            {args.limit} stories this run")
    print("=" * 70 + "\n")

    # Group models by their model name to run distinct models in parallel
    grouped_models = defaultdict(list)
    for model_config in models_config:
        model_key = model_config["model"]
        grouped_models[model_key].append(model_config)

    # Start concurrent tasks using ThreadPoolExecutor
    print(f"Launching speculative naming tasks concurrently (max {args.threads} parallel models)...")
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        for model_key, configs in grouped_models.items():
            futures.append(
                executor.submit(
                    process_model_group,
                    model_key=model_key,
                    group_configs=configs,
                    image_lists=image_lists,
                    reels_data=reels_data,
                    icons_dir=icons_dir,
                    session_factory=session_factory,
                    db_lock=db_lock,
                    db_path=db_path,
                    lambdas=args.lambdas,
                    hard_mode=args.hard,
                    hard_threshold=args.hard_threshold,
                    min_p_list=min_p_list,
                    intermediate_db_path=intermediate_db_path,
                )
            )

        # Wait for all thread pool tasks to finish
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error in model group task execution: {e}")

    print("\nSpeculative naming generation completed successfully!")

    # Flush Langfuse traces if configured
    lc.flush_langfuse()


if __name__ == "__main__":
    main()
