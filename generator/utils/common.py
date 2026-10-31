import os
import sys
import json
import random
import re
import uuid
import io
import base64
import math
import sqlite3
import threading
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import Any, List

from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Float, Integer, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

# Import langchain utils from the local utils folder
try:
    import utils.langchain as lc
except ImportError as e:
    try:
        from . import langchain as lc
    except ImportError:
        print(f"Failed to import utils.langchain in common.py: {e}")
        sys.exit(1)

# ------------------ ENVIRONMENT LOADING ------------------
def load_env_file(script_dir: Path):
    cwd_env = Path.cwd() / ".env"
    script_env = script_dir / ".env"
    
    for env_path in [cwd_env, script_env]:
        if env_path.exists():
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' in line:
                            key, val = line.split('=', 1)
                            key = key.strip()
                            val = val.strip()
                            # Strip quotes if present
                            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                                val = val[1:-1]
                            if key not in os.environ:
                                os.environ[key] = val
                print(f"[ENV] Loaded environment variables from {env_path}")
                return
            except Exception as e:
                print(f"[ENV] Error loading {env_path}: {e}")

# ------------------ DATABASE DEFINITIONS ------------------
BaseStories = declarative_base()

class ImageList(BaseStories):
    __tablename__ = "image_lists"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tags = Column(Text, nullable=False) # Store as JSON list string
    name = Column(String(255), nullable=True)
    image_data = Column(Text, nullable=True) # Base64 encoded PNG image
    created_at = Column(DateTime, default=datetime.utcnow)

class Story(BaseStories):
    __tablename__ = "stories"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_list_id = Column(String(36), ForeignKey("image_lists.id"), nullable=False)
    text = Column(Text, nullable=False)
    model_origin = Column(String(255), nullable=False)
    language = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    generation_type = Column(String(50), default="text")
    execution_time = Column(Float, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    prompt_modification = Column(String(100), nullable=True) # ID of prompt modification used
    think = Column(String(50), nullable=True) # Configured think value (e.g. true, false, or None)
    name_revisions = Column(Integer, default=0, nullable=True) # Number of name revisions needed
    raw_response = Column(Text, nullable=True) # Stores raw/JSON response if applicable
    names_match = Column(Boolean, nullable=True) # True if returned names match expected placeholders
    selected_names = Column(Text, nullable=True) # JSON: {"[placeholder]": "Name"} for multiphase/speculative

BaseReps = declarative_base()

class ModelNameCount(BaseReps):
    __tablename__ = "model_name_counts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_origin = Column(String(255), nullable=False)
    character_name = Column(String(255), nullable=False)
    count = Column(Integer, default=0, nullable=False)

# ------------------ DATABASE SETUP ------------------
def setup_stories_db(db_path: Path):
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False, "timeout": 30})
    BaseStories.metadata.create_all(bind=engine)
    
    from sqlalchemy import text
    
    # Enable WAL mode for better concurrency and prevent database locks
    try:
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.commit()
    except Exception as e:
        print(f"Database upgrade warning (WAL): {e}")

    # Upgrade database schema dynamically if columns are missing
    try:
        with engine.connect() as conn:
            cursor = conn.execute(text("PRAGMA table_info(stories)"))
            columns = [row[1] for row in cursor.fetchall()]
            
            if "prompt_modification" not in columns:
                print("Upgrading database: adding 'prompt_modification' column to 'stories' table...")
                conn.execute(text("ALTER TABLE stories ADD COLUMN prompt_modification VARCHAR(100)"))
                conn.commit()
                
            if "think" not in columns:
                print("Upgrading database: adding 'think' column to 'stories' table...")
                conn.execute(text("ALTER TABLE stories ADD COLUMN think VARCHAR(50)"))
                conn.commit()
                
            if "name_revisions" not in columns:
                print("Upgrading database: adding 'name_revisions' column to 'stories' table...")
                conn.execute(text("ALTER TABLE stories ADD COLUMN name_revisions INTEGER DEFAULT 0"))
                conn.commit()
                
            if "raw_response" not in columns:
                print("Upgrading database: adding 'raw_response' column to 'stories' table...")
                conn.execute(text("ALTER TABLE stories ADD COLUMN raw_response TEXT"))
                conn.commit()
                
            if "names_match" not in columns:
                print("Upgrading database: adding 'names_match' column to 'stories' table...")
                conn.execute(text("ALTER TABLE stories ADD COLUMN names_match BOOLEAN"))
                conn.commit()

            if "selected_names" not in columns:
                print("Upgrading database: adding 'selected_names' column to 'stories' table...")
                conn.execute(text("ALTER TABLE stories ADD COLUMN selected_names TEXT"))
                conn.commit()
                
            # Backfill existing stories based on model name containing "thinking" (case-insensitive)
            conn.execute(text("""
                UPDATE stories 
                SET think = 'true' 
                WHERE model_origin LIKE '%thinking%' AND (think IS NULL OR think = '')
            """))
            conn.execute(text("""
                UPDATE stories 
                SET think = 'false' 
                WHERE model_origin NOT LIKE '%thinking%' AND (think IS NULL OR think = '')
            """))
            conn.commit()
    except Exception as e:
        print(f"Database upgrade warning (stories): {e}")

    try:
        with engine.connect() as conn:
            cursor = conn.execute(text("PRAGMA table_info(image_lists)"))
            columns = [row[1] for row in cursor.fetchall()]
            if "image_data" not in columns:
                print("Upgrading database: adding 'image_data' column to 'image_lists' table...")
                conn.execute(text("ALTER TABLE image_lists ADD COLUMN image_data TEXT"))
                conn.commit()
    except Exception as e:
        print(f"Database upgrade warning (image_lists): {e}")

    Session = sessionmaker(bind=engine)
    return Session

def setup_reps_db(db_path: Path):
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False, "timeout": 30})
    BaseReps.metadata.create_all(bind=engine)
    
    from sqlalchemy import text
    
    try:
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.commit()
    except Exception as e:
        print(f"Repetitions database WAL upgrade warning: {e}")
        
    Session = sessionmaker(bind=engine)
    return Session

# ------------------ INTERMEDIATE CALLS DATABASE ------------------
_intermediate_db_lock = threading.Lock()

def setup_intermediate_db(db_path: Path):
    """Create the intermediate_calls table for logging multiphase pipeline data."""
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS intermediate_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_list_id TEXT,
            model_origin TEXT NOT NULL,
            prompt_modification TEXT,
            language TEXT NOT NULL,
            phase1_story TEXT,
            phase1_story_clean TEXT,
            placeholders TEXT,
            phase2_raw TEXT,
            phase2_parsed TEXT,
            selected_set TEXT,
            selected_index INTEGER,
            resolved_story TEXT,
            names_match BOOLEAN,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    conn.close()
    print(f"[Intermediate DB] Initialized at {db_path}")

def save_intermediate_call(
    db_path,
    image_list_id,
    model_origin,
    prompt_modification,
    language,
    phase1_story,
    phase1_story_clean,
    placeholders,
    phase2_raw,
    phase2_parsed,
    selected_set,
    selected_index,
    resolved_story,
    names_match,
):
    """Save one multiphase pipeline call to the intermediate_calls DB (thread-safe)."""
    with _intermediate_db_lock:
        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.execute("""
            INSERT INTO intermediate_calls (
                image_list_id, model_origin, prompt_modification, language,
                phase1_story, phase1_story_clean, placeholders,
                phase2_raw, phase2_parsed, selected_set, selected_index,
                resolved_story, names_match
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            image_list_id,
            model_origin,
            prompt_modification,
            language,
            phase1_story,
            phase1_story_clean,
            json.dumps(placeholders, ensure_ascii=False) if placeholders else None,
            phase2_raw,
            json.dumps(phase2_parsed, ensure_ascii=False) if phase2_parsed else None,
            json.dumps(selected_set, ensure_ascii=False) if selected_set else None,
            selected_index,
            resolved_story,
            names_match,
        ))
        conn.commit()
        conn.close()

# ------------------ CONFIG & FILE LOADING ------------------
def load_reels_data(reels_path: Path):
    try:
        with open(reels_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading reelsData.json from {reels_path}: {e}")
        sys.exit(1)

def load_models_config(models_config_path: Path):
    try:
        with open(models_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            if isinstance(config, list):
                return config
            else:
                return [config]
    except Exception as e:
        print(f"Error loading models config from {models_config_path}: {e}")
        sys.exit(1)

def load_prompt_content(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading prompt from {path}: {e}")
        return None

def load_model_instance(model_config: dict):
    api_key_name = model_config.get("api_key", "OLLAMA_APIKEY")
    api_key = os.getenv(api_key_name, api_key_name)
    think = model_config.get("think")
    timeout = model_config.get("timeout", 300)
    min_p = model_config.get("min_p")
    
    return lc.load_model(
        host=model_config.get("host") or model_config.get("base_url"),
        model=model_config["model"],
        apikey=api_key,
        json_schema=None,
        think=think,
        timeout=timeout,
        min_p=min_p
    )

def pick_language():
    langs = ["en", "fr", "de", "es"]
    weights = [0.4, 0.2, 0.2, 0.2]
    return random.choices(langs, weights=weights, k=1)[0]

# ------------------ UTILS & HELPER FUNCTIONS ------------------
def get_localized_names(images, lang):
    localized_names = []
    for img in images:
        localized_val = img.get(lang) or img.get("tag")
        localized_names.append(localized_val)
    return localized_names

def get_localized_names_from_tags(tags, reels_data, lang):
    localized_names = []
    for tag in tags:
        found = False
        for reel in reels_data:
            for img in reel["images"]:
                if img["tag"] == tag:
                    localized_val = img.get(lang) or img.get("tag")
                    localized_names.append(localized_val)
                    found = True
                    break
            if found:
                break
        if not found:
            localized_names.append(tag)
    return localized_names

def create_reel_strip(images, icons_dir: Path, icon_size=250):
    if not images:
        return ""
    try:
        strip_width = icon_size * len(images)
        strip_height = icon_size
        reel_strip = Image.new('RGBA', (strip_width, strip_height), (255, 255, 255, 0))

        for i, img in enumerate(images):
            icon_filename = img.get("url", "").split("/")[-1]
            if not icon_filename:
                continue
            path = icons_dir / icon_filename
            if not path.exists():
                print(f"Warning: Icon file not found: {path}")
                continue
            try:
                with Image.open(path) as pil_img:
                    img_resized = pil_img.convert('RGBA').resize((icon_size, icon_size), Image.LANCZOS)
                    x_offset = i * icon_size
                    reel_strip.paste(img_resized, (x_offset, 0), img_resized)
            except Exception as e:
                print(f"Error processing icon {path}: {e}")

        buffered = io.BytesIO()
        reel_strip.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Failed to create reel strip: {e}")
        return ""

def determine_think_string(model_config: dict, model_name: str) -> str:
    think_val = model_config.get("think")
    if isinstance(think_val, bool):
        return "true" if think_val else "false"
    elif isinstance(think_val, str):
        return think_val.lower()
    else:
        is_thinking = "thinking" in model_name.lower()
        return "true" if is_thinking else "false"

def extract_first_name(name_str: str) -> str:
    if not isinstance(name_str, str):
        return ""
    name_str = name_str.strip()
    
    # Strip common titles / honorifics (case-insensitive)
    titles_pattern = re.compile(
        r'^\s*(dr|mr|mrs|ms|prof|professor|sir|lady|lord|don|doña|herr|frau|dr\.|mr\.|mrs\.|ms\.|prof\.)\s+', 
        re.IGNORECASE
    )
    while True:
        replaced = titles_pattern.sub('', name_str)
        if replaced == name_str:
            break
        name_str = replaced
        
    # Split and take the first name part
    parts = name_str.split()
    if not parts:
        return ""
        
    first_name = parts[0].strip().title()
    # Remove punctuation/symbols that could cling to characters
    first_name = re.sub(r'[^a-zA-Z\u00C0-\u017F]', '', first_name)
    return first_name

# ------------------ STORY GENERATION WORKFLOWS ------------------
def generate_story(
    model_config: dict,
    language: str,
    model_type: str = "chat", # "chat" or "vision"
    tags: list = None,
    localized_names: list = None,
    image_data: str = None,
    mime_type: str = "image/png",
    prompt_modification_text: str = None,
    without_stimulus: bool = False,
    script_dir: Path = None
):
    if script_dir is None:
        script_dir = Path(__file__).resolve().parent.parent

    model = load_model_instance(model_config)
    if not model:
        raise RuntimeError(f"Failed to load model: {model_config.get('name', model_config['model'])}")

    if without_stimulus:
        # System prompt path
        system_prompt_path = script_dir / "prompts" / "story_without_stimulus" / "system" / f"story_generation_{language}.txt"
        if not system_prompt_path.exists():
            system_prompt_path = script_dir / "prompts" / "story_without_stimulus" / "system" / "story_generation_es.txt"
        system_prompt = load_prompt_content(system_prompt_path) or "Eres un escritor creativo de historias."

        # Load prompt without stimulus
        prompt_path = script_dir / "prompts" / "story_without_stimulus" / f"chat_{language}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Required prompt file not found: {prompt_path}")
            
        prompt = load_prompt_content(prompt_path)
    else:
        # System prompt path
        system_prompt_path = script_dir / "prompts" / "system" / f"story_generation_{language}.txt"
        if not system_prompt_path.exists():
            system_prompt_path = script_dir / "prompts" / "system" / "story_generation_es.txt"
        system_prompt = load_prompt_content(system_prompt_path) or "Eres un escritor creativo de historias."

        # User prompt path
        prompt_type = "vision" if model_type == "vision" else "chat"
        prompt_path = script_dir / "prompts" / "story_generation" / f"{prompt_type}_{language}.txt"
        if not prompt_path.exists():
            prompt_path = script_dir / "prompts" / "story_generation" / f"{prompt_type}_es.txt"
        prompt_template = load_prompt_content(prompt_path) or "Escribe una historia con: {image_names}"

        image_names_text = ", ".join(localized_names if localized_names else (tags if tags else []))
        prompt = prompt_template.format(image_names=image_names_text)

    # Append prompt modification text if provided
    if prompt_modification_text:
        prompt = f"{prompt}\n{prompt_modification_text}"

    supports_system = model_config.get("supports_system_instructions", True)

    if image_data and model_type == "vision" and not without_stimulus:
        response, metrics = lc.generate_multimodal_response(
            prompt_text=prompt,
            system_prompt=system_prompt,
            model=model,
            image_data=image_data,
            mime_type=mime_type,
            supports_system_instructions=supports_system
        )
    else:
        response, metrics = lc.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            supports_system_instructions=supports_system
        )

    mode_suffix = "vision" if (image_data and model_type == "vision" and not without_stimulus) else "chat"
    final_model_name = f"{model_config.get('name', model_config['model'])} ({mode_suffix})"
    
    return response, final_model_name, metrics

def generate_multiphase_story(
    model_config: dict,
    model_type: str, # "chat" or "vision"
    tags: list,
    language: str,
    localized_names: list,
    image_data: str = None,
    mime_type: str = "image/png",
    script_dir: Path = None
):
    if script_dir is None:
        script_dir = Path(__file__).resolve().parent.parent
    model = load_model_instance(model_config)
    if not model:
        raise RuntimeError(f"Failed to load model: {model_config.get('name', model_config['model'])}")
    # Set up Langfuse callback handler for the multiphase trace if enabled
    trace = None
    span1 = None
    handler1 = None
    langfuse_client = lc.get_langfuse_client()
    if langfuse_client:
        try:
            model_name = model_config.get("name", model_config["model"])
            min_p = model_config.get("min_p")
            trace = langfuse_client.trace(
                name=f"Multiphase-Story-Generation-{model_name}",
                metadata={
                    "model": model_name,
                    "language": language,
                    "min_p": min_p,
                    "model_type": model_type,
                    "tags": tags
                },
                tags=["multiphase"]
            )
            span1 = trace.span(name="Phase-1-Draft-Story")
            from langfuse.langchain import CallbackHandler
            handler1 = CallbackHandler(trace_context={"trace_id": trace.id, "parent_span_id": span1.id})
        except Exception as e:
            print(f"Warning: Failed to create Langfuse trace/span: {e}")

    config1 = {"callbacks": [handler1]} if handler1 else None

    # ------------------ PHASE 1: GENERATE STORY WITH PLACEHOLDERS ------------------
    # Load Phase 1 prompt
    prompt_type = "vision" if model_type == "vision" else "chat"
    p1_prompt_path = script_dir / "prompts" / "story_generation_multiphase" / f"phase1_{prompt_type}_{language}.txt"
    if not p1_prompt_path.exists():
        p1_prompt_path = script_dir / "prompts" / "story_generation_multiphase" / f"phase1_{prompt_type}_es.txt"
    p1_template = load_prompt_content(p1_prompt_path)
    if not p1_template:
        raise RuntimeError(f"Failed to load Phase 1 prompt: {p1_prompt_path}")

    image_names_text = ", ".join(localized_names if localized_names else tags)
    p1_prompt = p1_template.format(image_names=image_names_text)

    # Load base system prompt
    sys_prompt_path = script_dir / "prompts" / "system" / f"story_generation_{language}.txt"
    if not sys_prompt_path.exists():
        sys_prompt_path = script_dir / "prompts" / "system" / "story_generation_es.txt"
    system_prompt = load_prompt_content(sys_prompt_path) or "Eres un escritor creativo de historias."

    # Append Phase 1 negative naming instruction to system prompt
    naming_instruction = {
        "es": "\n\nIMPORTANTE: No asignes nombres propios a los personajes bajo ninguna circunstancia. En su lugar, utiliza marcadores de posición genéricos entre corchetes como [personaje_1], [personaje_femenino_1], [robot_1], etc.",
        "en": "\n\nIMPORTANT: Do not assign proper names to characters. Instead, use generic bracketed placeholders like [character_1], [female_character_1], [robot_1], etc.",
        "fr": "\n\nIMPORTANT : N'attribuez pas de noms propres aux personnages. Utilisez plutôt des marqueurs de position génériques entre crochets comme [personnage_1], [personnage_feminin_1], [robot_1], etc.",
        "de": "\n\nWICHTIG: Weise den Charakteren keine Eigennamen zu. Verwende stattdessen generische Platzhalter in eckigen Klammern wie [charakter_1], [weiblicher_charakter_1], [roboter_1] usw."
    }
    system_prompt += naming_instruction.get(language, naming_instruction["es"])

    supports_system = model_config.get("supports_system_instructions", True)

    print(f"   [Phase 1] Launching Phase 1 generation...")
    if image_data and model_type == "vision":
        p1_story, p1_metrics = lc.generate_multimodal_response(
            prompt_text=p1_prompt,
            system_prompt=system_prompt,
            model=model,
            image_data=image_data,
            mime_type=mime_type,
            supports_system_instructions=supports_system,
            config=config1
        )
    else:
        p1_story, p1_metrics = lc.generate_response(
            prompt=p1_prompt,
            system_prompt=system_prompt,
            model=model,
            supports_system_instructions=supports_system,
            config=config1
        )

    if span1:
        try:
            span1.end()
        except Exception as e:
            print(f"Warning: Failed to end Phase 1 span: {e}")

    if not p1_story:
        return None, None, {}

    print(f"   [Phase 1] Completed Phase 1. Draft story length: {len(p1_story)} chars.")

    # ------------------ PHASE 2: REPLACE PLACEHOLDERS WITH CREATIVE NAMES ------------------
    # Load Phase 2 prompt
    p2_prompt_path = script_dir / "prompts" / "story_generation_multiphase" / f"phase2_chat_{language}.txt"
    if not p2_prompt_path.exists():
        p2_prompt_path = script_dir / "prompts" / "story_generation_multiphase" / f"phase2_chat_es.txt"
    p2_template = load_prompt_content(p2_prompt_path)
    if not p2_template:
        raise RuntimeError(f"Failed to load Phase 2 prompt: {p2_prompt_path}")

    p2_prompt = p2_template.format(placeholder_story=p1_story)

    # Phase 2 System Prompt
    p2_sys_instruction = {
        "es": "Eres un editor literario experto. Debes reemplazar los marcadores de posición entre corchetes con nombres creativos y originales, manteniendo el resto de la historia idéntica.",
        "en": "You are an expert literary editor. You must replace the bracketed placeholders with creative and original names, keeping the rest of the story identical.",
        "fr": "Vous êtes un éditeur littéraire expert. Vous devez remplacer les marqueurs de position entre crochets par des noms créatifs et originaux, en gardant le reste de l'histoire identique.",
        "de": "Du bist ein erfahrener Lektor. Du musst die Platzhalter in eckigen Klammern durch kreative und originelle Namen ersetzen und den Rest der Geschichte identisch halten."
    }
    p2_system_prompt = p2_sys_instruction.get(language, p2_sys_instruction["es"])

    # Set up Phase 2 Langfuse callback handler
    span2 = None
    handler2 = None
    if trace:
        try:
            span2 = trace.span(name="Phase-2-Naming-Replacement")
            from langfuse.langchain import CallbackHandler
            handler2 = CallbackHandler(trace_context={"trace_id": trace.id, "parent_span_id": span2.id})
        except Exception as e:
            print(f"Warning: Failed to create Phase 2 span: {e}")

    config2 = {"callbacks": [handler2]} if handler2 else None

    print(f"   [Phase 2] Launching Phase 2 naming replacement...")
    p2_story, p2_metrics = lc.generate_response(
        prompt=p2_prompt,
        system_prompt=p2_system_prompt,
        model=model,
        supports_system_instructions=supports_system,
        config=config2
    )

    if span2:
        try:
            span2.end()
        except Exception as e:
            print(f"Warning: Failed to end Phase 2 span: {e}")

    if not p2_story:
        print("   [Phase 2] Warning: Phase 2 naming returned empty story. Skipping save.")
        return None, None, {}

    print(f"   [Phase 2] Completed Phase 2. Final story length: {len(p2_story)} chars.")

    # Combine metrics
    combined_metrics = {
        "execution_time": p1_metrics.get("execution_time", 0.0) + p2_metrics.get("execution_time", 0.0),
        "input_tokens": (p1_metrics.get("input_tokens") or 0) + (p2_metrics.get("input_tokens") or 0),
        "output_tokens": (p1_metrics.get("output_tokens") or 0) + (p2_metrics.get("output_tokens") or 0)
    }

    mode_suffix = "vision" if (image_data and model_type == "vision") else "chat"
    final_model_name = f"{model_config.get('name', model_config['model'])} ({mode_suffix})"

    return p2_story, final_model_name, combined_metrics

def normalize_placeholder_key(s):
    if not isinstance(s, str):
        return ""
    # Convert to lowercase
    s = s.lower()
    # Strip brackets
    s = s.replace('[', '').replace(']', '')
    # Remove underscores, hyphens, spaces
    s = re.sub(r'[^a-z0-9]', '', s)
    # Map common variations
    s = s.replace('personnage', 'char')
    s = s.replace('personaje', 'char')
    s = s.replace('character', 'char')
    s = s.replace('person', 'char')
    s = s.replace('pers', 'char')
    s = s.replace('feminin', 'fem')
    s = s.replace('femenino', 'fem')
    s = s.replace('female', 'fem')
    return s

def find_lowest_probability_set(name_sets):
    if not name_sets or not isinstance(name_sets, list):
        return None
    
    lowest_set = None
    min_prob = float('inf')
    
    for ns in name_sets:
        if not isinstance(ns, dict):
            continue
        prob = ns.get("probability", ns.get("Probability"))
        if prob is None:
            continue
        try:
            prob_val = float(prob)
            if prob_val < min_prob:
                min_prob = prob_val
                lowest_set = ns
        except (ValueError, TypeError):
            continue
            
    if lowest_set is None and name_sets:
        for ns in name_sets:
            if isinstance(ns, dict) and ("names" in ns or "Names" in ns):
                lowest_set = ns
                break
                
    return lowest_set


class SlopTracker:
    """Per-cell dynamic slop tracker with SQLite persistence.

    Initialized from baseline fingerprint, accumulates runtime name usage.
    slop_score(name) = max(baseline_freq, runtime_freq if total > COLD_START else 0)

    Each cell (model_origin, prompt_modification) gets its own tracker,
    starting clean from the baseline fingerprint. As names are used during
    generation, their runtime frequency rises and eventually exceeds the
    baseline — triggering the exp(-lambda * slop) penalty even for names
    not in the original fingerprint (e.g., "Kaelen" in Qwen 3.5 9B).
    """

    COLD_START = 50  # min total names before runtime_freq kicks in

    def __init__(self, model_origin, prompt_modification, baseline_freqs, db_path=None):
        self.model_origin = model_origin
        self.prompt_modification = prompt_modification
        self._baseline = dict(baseline_freqs) if baseline_freqs else {}
        self._runtime = Counter()
        self._total = 0
        self._lock = threading.Lock()
        self._db_path = str(db_path) if db_path else None
        if self._db_path:
            self._setup_db()
            self._load()

    @staticmethod
    def _normalize(raw_name):
        if not raw_name or not isinstance(raw_name, str):
            return ""
        n = raw_name.strip().lower()
        n = "".join(
            c for c in unicodedata.normalize("NFD", n)
            if unicodedata.category(c) != "Mn"
        )
        if not n or len(n) < 2:
            return ""
        return n.split()[0] if n.split() else n

    def _setup_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS slop_runtime_tracker (
                model_origin TEXT NOT NULL,
                prompt_modification TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (model_origin, prompt_modification, normalized_name)
            )
        """)
        conn.commit()
        conn.close()

    def _load(self):
        conn = sqlite3.connect(self._db_path)
        cur = conn.execute(
            "SELECT normalized_name, count FROM slop_runtime_tracker "
            "WHERE model_origin = ? AND prompt_modification = ?",
            (self.model_origin, self.prompt_modification),
        )
        with self._lock:
            for name, count in cur.fetchall():
                self._runtime[name] = count
                self._total += count
        conn.close()
        if self._total > 0:
            print(f"   [Tracker] Loaded {self._total} runtime counts for "
                  f"{self.model_origin}/{self.prompt_modification}")

    def slop_score(self, raw_name):
        """Combined slop score: max(baseline_freq, runtime_freq)."""
        norm = self._normalize(raw_name)
        if not norm:
            return 0.0
        baseline = self._baseline.get(norm, 0.0)
        with self._lock:
            if self._total > self.COLD_START:
                runtime = self._runtime.get(norm, 0) / self._total
            else:
                runtime = 0.0
        return max(baseline, runtime)

    def record(self, names):
        """Record that these names were selected in a story."""
        with self._lock:
            for raw_name in names:
                norm = self._normalize(raw_name)
                if norm:
                    self._runtime[norm] += 1
                    self._total += 1

    def save(self):
        """Persist runtime counts to SQLite."""
        if not self._db_path:
            return
        with self._lock:
            data = list(self._runtime.items())
        if not data:
            return
        conn = sqlite3.connect(self._db_path)
        conn.executemany(
            "INSERT INTO slop_runtime_tracker "
            "(model_origin, prompt_modification, normalized_name, count) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(model_origin, prompt_modification, normalized_name) "
            "DO UPDATE SET count = ?",
            [(self.model_origin, self.prompt_modification, name, count, count)
             for name, count in data],
        )
        conn.commit()
        conn.close()

    def stats(self):
        """Return summary stats for logging."""
        with self._lock:
            return {
                "total": self._total,
                "unique": len(self._runtime),
                "top5": self._runtime.most_common(5),
            }


def select_speculative_set(
    name_sets,
    tracker: SlopTracker,
    lambda_value: float = 10.0,
    hard_mode: bool = False,
    hard_threshold: float = 0.01,
):
    """
    Speculative naming selection with dynamic slop tracking.

    Uses a SlopTracker (initialized from baseline fingerprint, accumulates
    runtime usage) to steer selection away from overused names.

    Two modes:
      - REJECTION (lambda_value): accept_prob(i) ∝ p_i × prod(exp(-λ × slop_score))
        Samples a set according to these weights. Preserves the model's
        distribution while penalizing slop. lambda_value=0 → pure verbalized.
      - HARD (hard_mode): accept first set where ALL names have slop_score <
        hard_threshold. Falls back to least-slop set if none pass.

    After selecting a set, records the chosen names in the tracker so that
    future selections in the same cell see updated slop frequencies.

    Args:
        name_sets: list of {"probability": float, "names": {placeholder: name}}
        tracker: SlopTracker for this cell (model × prompt_modification)
        lambda_value: λ for exponential penalty (ignored if hard_mode)
        hard_mode: if True, use threshold-based selection instead of λ
        hard_threshold: slop frequency threshold for hard mode

    Returns:
        Selected name set dict, or None if input is empty.
    """
    if not name_sets or not isinstance(name_sets, list):
        return None

    slop_score = tracker.slop_score

    # ── HARD mode: threshold-based greedy accept ──────────────────────────
    if hard_mode:
        # Sort by probability descending so HARD prefers the highest-probability clean set
        sorted_sets = sorted(
            [ns for ns in name_sets if isinstance(ns, dict)],
            key=lambda ns: float(ns.get("probability", ns.get("Probability", 0))),
            reverse=True,
        )
        for ns in sorted_sets:
            names_dict = ns.get("names", ns.get("Names"))
            if not names_dict or not isinstance(names_dict, dict):
                continue
            if all(slop_score(v) < hard_threshold for v in names_dict.values() if v):
                selected = ns
                break
        else:
            # Fallback: return the set with minimum total slop_score
            best_ns = None
            best_slop = float("inf")
            for ns in sorted_sets:
                names_dict = ns.get("names", ns.get("Names"))
                if not names_dict or not isinstance(names_dict, dict):
                    continue
                total_slop = sum(slop_score(v) for v in names_dict.values() if v)
                if total_slop < best_slop:
                    best_slop = total_slop
                    best_ns = ns
            selected = best_ns
    else:
        # ── REJECTION mode: weighted sampling with exp(-λ × slop) ──────────
        weights = []
        for ns in name_sets:
            if not isinstance(ns, dict):
                weights.append(0.0)
                continue
            try:
                p = float(ns.get("probability", ns.get("Probability", 0.1)))
            except (ValueError, TypeError):
                p = 0.1
            p = max(p, 1e-6)

            names_dict = ns.get("names", ns.get("Names"))
            penalty = 1.0
            if names_dict and isinstance(names_dict, dict):
                for raw_name in names_dict.values():
                    if raw_name:
                        s = slop_score(raw_name)
                        penalty *= math.exp(-lambda_value * s)
            weights.append(p * penalty)

        total = sum(weights)
        if total <= 0:
            return random.choice(name_sets) if name_sets else None

        # Weighted random choice
        r = random.random() * total
        cumulative = 0.0
        selected = name_sets[-1]
        for ns, w in zip(name_sets, weights):
            cumulative += w
            if r <= cumulative:
                selected = ns
                break

    # Record selected names in the tracker for future calls
    if selected and isinstance(selected, dict):
        names_dict = selected.get("names", selected.get("Names"))
        if names_dict and isinstance(names_dict, dict):
            tracker.record(list(names_dict.values()))

    return selected


def generate_multiphase_prob_story(
    model_config: dict,
    model_type: str, # "chat" or "vision"
    tags: list,
    language: str,
    localized_names: list,
    image_data: str = None,
    mime_type: str = "image/png",
    script_dir: Path = None,
    selection_fn=None,  # Optional override for name set selection (speculative naming)
    intermediate_db_path=None,  # Path to intermediate_calls DB (enables logging)
    intermediate_meta=None,  # dict with image_list_id, prompt_modification
):
    if script_dir is None:
        script_dir = Path(__file__).resolve().parent.parent

    model = load_model_instance(model_config)
    if not model:
        raise RuntimeError(f"Failed to load model: {model_config.get('name', model_config['model'])}")

    # Set up Langfuse callback handler for the multiphase trace if enabled
    trace = None
    span1 = None
    handler1 = None
    langfuse_client = lc.get_langfuse_client()
    if langfuse_client:
        try:
            model_name = model_config.get("name", model_config["model"])
            min_p = model_config.get("min_p")
            trace = langfuse_client.trace(
                name=f"Multiphase-Story-Generation-Prob-{model_name}",
                metadata={
                    "model": model_name,
                    "language": language,
                    "min_p": min_p,
                    "model_type": model_type,
                    "tags": tags
                },
                tags=["multiphase_prob"]
            )
            span1 = trace.span(name="Phase-1-Draft-Story")
            from langfuse.langchain import CallbackHandler
            handler1 = CallbackHandler(trace_context={"trace_id": trace.id, "parent_span_id": span1.id})
        except Exception as e:
            print(f"Warning: Failed to create Langfuse trace/span: {e}")

    config1 = {"callbacks": [handler1]} if handler1 else None

    # ------------------ PHASE 1: GENERATE STORY WITH PLACEHOLDERS ------------------
    # Load Phase 1 prompt
    prompt_type = "vision" if model_type == "vision" else "chat"
    p1_prompt_path = script_dir / "prompts" / "verbalized_sampling" / f"phase1_{prompt_type}_{language}.txt"
    if not p1_prompt_path.exists():
        p1_prompt_path = script_dir / "prompts" / "verbalized_sampling" / f"phase1_{prompt_type}_es.txt"
    p1_template = load_prompt_content(p1_prompt_path)
    if not p1_template:
        raise RuntimeError(f"Failed to load Phase 1 prompt: {p1_prompt_path}")

    image_names_text = ", ".join(localized_names if localized_names else tags)
    p1_prompt = p1_template.format(image_names=image_names_text)

    # Load base system prompt
    sys_prompt_path = script_dir / "prompts" / "system" / f"story_generation_{language}.txt"
    if not sys_prompt_path.exists():
        sys_prompt_path = script_dir / "prompts" / "system" / "story_generation_es.txt"
    system_prompt = load_prompt_content(sys_prompt_path) or "Eres un escritor creativo de historias."

    # Append Phase 1 negative naming instruction to system prompt
    naming_instruction = {
        "es": "\n\nIMPORTANTE: No asignes nombres propios a los personajes bajo ninguna circunstancia. En su lugar, utiliza marcadores de posición genéricos y claros entre corchetes como [personaje_1], [personaje_femenino_1], [robot_1], etc.",
        "en": "\n\nIMPORTANT: Do not assign proper names to characters. Instead, use generic bracketed placeholders like [character_1], [female_character_1], [robot_1], etc.",
        "fr": "\n\nIMPORTANT : N'attribuez pas de noms propres aux personnages. Utilisez plutôt des marqueurs de position génériques entre crochets comme [personnage_1], [personnage_feminin_1], [robot_1], etc.",
        "de": "\n\nWICHTIG: Weise den Charakteren keine Eigennamen zu. Verwende stattdessen generische Platzhalter in eckigen Klammern wie [charakter_1], [weiblicher_charakter_1], [roboter_1] usw."
    }
    system_prompt += naming_instruction.get(language, naming_instruction["es"])

    supports_system = model_config.get("supports_system_instructions", True)

    print(f"   [Phase 1] Launching Phase 1 generation...")
    if image_data and model_type == "vision":
        p1_story, p1_metrics = lc.generate_multimodal_response(
            prompt_text=p1_prompt,
            system_prompt=system_prompt,
            model=model,
            image_data=image_data,
            mime_type=mime_type,
            supports_system_instructions=supports_system,
            config=config1
        )
    else:
        p1_story, p1_metrics = lc.generate_response(
            prompt=p1_prompt,
            system_prompt=system_prompt,
            model=model,
            supports_system_instructions=supports_system,
            config=config1
        )

    if span1:
        try:
            span1.end()
        except Exception as e:
            print(f"Warning: Failed to end Phase 1 span: {e}")

    if not p1_story:
        return None, None, False, None, {}

    print(f"   [Phase 1] Completed Phase 1. Draft story length: {len(p1_story)} chars.")

    # ------------------ PHASE 2: REPLACE PLACEHOLDERS WITH VERBALIZED SAMPLING NAMES ------------------
    # Detect placeholders in p1_story
    raw_placeholders = re.findall(r'(\[[a-zA-Z0-9_\u00C0-\u017F ]+\])', p1_story)

    # --- Fix 1: Filter out brackets that wrap image object names instead of characters ---
    # Build a set of normalised image-name tokens from the localized names list
    def _norm_image_token(s: str) -> str:
        """Lowercase, strip brackets, collapse spaces/underscores/hyphens."""
        return re.sub(r'[^a-z0-9\u00c0-\u017f]', '', s.lower())

    image_name_tokens = set()
    for name in (localized_names if localized_names else tags):
        # Each multi-word name (e.g. "Open book") adds tokens for the whole phrase
        # AND for each individual word longer than 2 chars
        full_tok = _norm_image_token(name)
        if full_tok:
            image_name_tokens.add(full_tok)
        for word in re.split(r'[\s_]+', name):
            wt = _norm_image_token(word)
            if len(wt) > 2:
                image_name_tokens.add(wt)

    def _is_object_placeholder(ph: str) -> bool:
        """Return True if placeholder inner text matches an image object name."""
        inner = _norm_image_token(ph[1:-1])  # strip [ and ]
        if not inner:
            return False
        # Full match OR the inner text contains any image-name token
        if inner in image_name_tokens:
            return True
        # Partial substring match (object name is fully contained in placeholder text)
        for tok in image_name_tokens:
            if tok and (tok in inner or inner in tok):
                return True
        return False

    object_placeholders = [ph for ph in set(raw_placeholders) if _is_object_placeholder(ph)]
    character_placeholders_raw = [ph for ph in set(raw_placeholders) if not _is_object_placeholder(ph)]

    # Remove brackets from object-name placeholders in the story text
    p1_story_clean = p1_story
    for op in object_placeholders:
        inner_text = op[1:-1]  # e.g. "barco"
        p1_story_clean = p1_story_clean.replace(op, inner_text)
    if object_placeholders:
        print(f"   [Phase 1] Removed {len(object_placeholders)} object-name bracket(s): {object_placeholders}")

    # --- Fix 2: Case-insensitive deduplication — prefer lowercase variant ---
    seen_lower: dict = {}
    for ph in character_placeholders_raw:
        key = ph.lower()
        if key not in seen_lower:
            seen_lower[key] = ph.lower()  # store lowercase canonical form
        # If a different-case variant exists in the story, normalise it to lowercase
    # Apply canonical lowercase forms to the cleaned story
    for ph in character_placeholders_raw:
        canonical = seen_lower[ph.lower()]
        if ph != canonical:
            p1_story_clean = p1_story_clean.replace(ph, canonical)

    placeholders = sorted(set(seen_lower.values()))

    if not placeholders:
        print("   [Phase 2] Warning: No placeholders found in Phase 1 story. Returning draft story.")
        combined_metrics = {
            "execution_time": p1_metrics.get("execution_time", 0.0),
            "input_tokens": p1_metrics.get("input_tokens", 0),
            "output_tokens": p1_metrics.get("output_tokens", 0)
        }
        mode_suffix = "vision" if (image_data and model_type == "vision") else "chat"
        final_model_name = f"{model_config.get('name', model_config['model'])} ({mode_suffix})"
        raw_response_obj = {
            "story_draft": p1_story,
            "raw_names_response": None
        }
        raw_response_str = json.dumps(raw_response_obj, ensure_ascii=False, indent=2)
        return p1_story_clean, raw_response_str, True, final_model_name, combined_metrics

    # Load Phase 2 prompt (Verbalized Sampling)
    p2_prompt_path = script_dir / "prompts" / "verbalized_sampling" / f"phase2_chat_{language}.txt"
    if not p2_prompt_path.exists():
        p2_prompt_path = script_dir / "prompts" / "verbalized_sampling" / f"phase2_chat_es.txt"
    p2_template = load_prompt_content(p2_prompt_path)
    if not p2_template:
        raise RuntimeError(f"Failed to load Phase 2 prompt: {p2_prompt_path}")

    # Generate a dynamic example JSON format using the actual placeholders found in the story
    name_label = "nombre"
    if language == "en" or language == "de":
        name_label = "name"
    elif language == "fr":
        name_label = "nom"

    sample_names = {}
    for i, ph in enumerate(placeholders):
        sample_names[ph] = f"{name_label}_{i+1}"

    sample_set = {
        "probability": 0.35,
        "names": sample_names
    }
    sample_set_str = json.dumps(sample_set, indent=2, ensure_ascii=False)
    indented_sample_set = "\n".join("    " + line for line in sample_set_str.splitlines())
    sample_json_str = f"{{\n  \"name_sets\": [\n{indented_sample_set},\n    ...\n  ]\n}}"

    placeholders_list = ", ".join(placeholders)
    p2_prompt = p2_template.replace("{placeholder_story}", p1_story_clean)\
                            .replace("{placeholders_list}", placeholders_list)\
                            .replace("{sample_json}", sample_json_str)

    # Phase 2 System Prompt
    p2_sys_instruction = {
        "es": "Eres un editor literario experto. Debes generar asignaciones de nombres para los personajes en formato JSON.",
        "en": "You are an expert literary editor. You must generate character name assignments in JSON format.",
        "fr": "Vous êtes un éditeur littéraire expert. Vous devez générer des attributions de noms de personnages au format JSON.",
        "de": "Du bist ein erfahrener Lektor. Du musst Charakter-Namenszuweisungen im JSON-Format generieren."
    }
    p2_system_prompt = p2_sys_instruction.get(language, p2_sys_instruction["es"])

    # Set up Phase 2 Langfuse callback handler
    span2 = None
    handler2 = None
    if trace:
        try:
            span2 = trace.span(name="Phase-2-Verbalized-Sampling")
            from langfuse.langchain import CallbackHandler
            handler2 = CallbackHandler(trace_context={"trace_id": trace.id, "parent_span_id": span2.id})
        except Exception as e:
            print(f"Warning: Failed to create Phase 2 span: {e}")

    config2 = {"callbacks": [handler2]} if handler2 else None

    print(f"   [Phase 2] Launching Phase 2 Verbalized Sampling name generation...")
    p2_story, p2_metrics = lc.generate_response(
        prompt=p2_prompt,
        system_prompt=p2_system_prompt,
        model=model,
        supports_system_instructions=supports_system,
        config=config2
    )

    if span2:
        try:
            span2.end()
        except Exception as e:
            print(f"Warning: Failed to end Phase 2 span: {e}")

    if not p2_story:
        print("   [Phase 2] Warning: Phase 2 verbalized sampling returned empty response.")
        combined_metrics = {
            "execution_time": p1_metrics.get("execution_time", 0.0),
            "input_tokens": (p1_metrics.get("input_tokens") or 0),
            "output_tokens": (p1_metrics.get("output_tokens") or 0)
        }
        mode_suffix = "vision" if (image_data and model_type == "vision") else "chat"
        final_model_name = f"{model_config.get('name', model_config['model'])} ({mode_suffix})"
        raw_response_obj = {
            "story_draft": p1_story,
            "raw_names_response": None
        }
        raw_response_str = json.dumps(raw_response_obj, ensure_ascii=False, indent=2)
        return p1_story_clean, raw_response_str, False, final_model_name, combined_metrics

    print(f"   [Phase 2] Completed Phase 2. Response length: {len(p2_story)} chars.")

    # Parse Phase 2 response using robust_parse
    try:
        from utils.robust_parse import parse_with_repair
    except ImportError:
        from .robust_parse import parse_with_repair

    parsed_data, diag = parse_with_repair(p2_story)

    name_sets = []
    if parsed_data and isinstance(parsed_data, dict):
        if "name_sets" in parsed_data:
            name_sets = parsed_data["name_sets"]
        elif "nameSets" in parsed_data:
            name_sets = parsed_data["nameSets"]

    # Combine metrics
    combined_metrics = {
        "execution_time": p1_metrics.get("execution_time", 0.0) + p2_metrics.get("execution_time", 0.0),
        "input_tokens": (p1_metrics.get("input_tokens") or 0) + (p2_metrics.get("input_tokens") or 0),
        "output_tokens": (p1_metrics.get("output_tokens") or 0) + (p2_metrics.get("output_tokens") or 0)
    }

    mode_suffix = "vision" if (image_data and model_type == "vision") else "chat"
    final_model_name = f"{model_config.get('name', model_config['model'])} ({mode_suffix})"

    # Perform name replacement using the selected name set
    resolved_story = p1_story_clean
    names_match = False
    lowest_set = None
    selected_index = None
    if name_sets:
        if selection_fn is not None:
            lowest_set = selection_fn(name_sets)
        else:
            lowest_set = find_lowest_probability_set(name_sets)
        if lowest_set is not None:
            for i, ns in enumerate(name_sets):
                if ns is lowest_set:
                    selected_index = i
                    break
        if lowest_set:
            names_dict = lowest_set.get("names", lowest_set.get("Names"))
            if names_dict and isinstance(names_dict, dict):
                # Check if the returned list of names matches the expected list of placeholders
                expected_norm = {normalize_placeholder_key(p) for p in placeholders}
                returned_norm = {normalize_placeholder_key(k) for k in names_dict.keys()}
                names_match = (expected_norm == returned_norm)

                # 1. Group names by their normalized placeholder key
                norm_to_name = {}
                for k, v in names_dict.items():
                    if v:
                        norm_k = normalize_placeholder_key(k)
                        # Keep the longest name if there are multiple keys normalizing to the same value
                        if norm_k not in norm_to_name or len(str(v)) > len(str(norm_to_name[norm_k])):
                            norm_to_name[norm_k] = v
                
                # 2. Group the actual story placeholders by their normalized key
                ph_by_norm = {}
                for ph in placeholders:
                    norm_ph = normalize_placeholder_key(ph)
                    if norm_ph not in ph_by_norm:
                        ph_by_norm[norm_ph] = []
                    ph_by_norm[norm_ph].append(ph)

                # 3. First pass: Exact normalized match
                matched_norms = set()
                for norm_k, val in norm_to_name.items():
                    if norm_k in ph_by_norm:
                        for ph in ph_by_norm[norm_k]:
                            resolved_story = resolved_story.replace(ph, str(val))
                        matched_norms.add(norm_k)
                
                # Keep track of remaining unmatched placeholders and names
                remaining_norms = [n for n in ph_by_norm if n not in matched_norms]
                remaining_names = {k: v for k, v in norm_to_name.items() if k not in matched_norms}
                
                # Second pass: match by number suffix if present (e.g. if both have '1' or '2')
                for norm_ph in list(remaining_norms):
                    digits = []
                    for ph in ph_by_norm[norm_ph]:
                        d = re.findall(r'\d+', ph)
                        if d:
                            digits.extend(d)
                    if digits:
                        num = digits[0]
                        for norm_name_k, val in list(remaining_names.items()):
                            name_digits = re.findall(r'\d+', norm_name_k)
                            if name_digits and name_digits[0] == num:
                                for ph in ph_by_norm[norm_ph]:
                                    resolved_story = resolved_story.replace(ph, str(val))
                                remaining_names.pop(norm_name_k)
                                remaining_norms.remove(norm_ph)
                                break
                
                # Third pass: if only one remaining placeholder and one remaining name, match them
                if len(remaining_norms) == 1 and len(remaining_names) == 1:
                    norm_ph = remaining_norms[0]
                    val = list(remaining_names.values())[0]
                    for ph in ph_by_norm[norm_ph]:
                        resolved_story = resolved_story.replace(ph, str(val))
                    remaining_norms.clear()
                    remaining_names.clear()
                
                # Fourth pass: positional alignment fallback
                if remaining_norms and remaining_names:
                    remaining_norms.sort()
                    sorted_remaining_names_keys = sorted(remaining_names.keys())
                    for i in range(min(len(remaining_norms), len(sorted_remaining_names_keys))):
                        norm_ph = remaining_norms[i]
                        val = remaining_names[sorted_remaining_names_keys[i]]
                        for ph in ph_by_norm[norm_ph]:
                            resolved_story = resolved_story.replace(ph, str(val))
    else:
        print(f"   [Phase 2] Warning: Failed to parse valid name sets JSON from response. Diagnostics: {diag}")

    raw_response_obj = {
        "story_draft": p1_story,
        "raw_names_response": p2_story,
        "placeholders": placeholders,
        "name_sets_parsed": name_sets,
        "selected_set": lowest_set,
        "selected_index": selected_index,
    }
    raw_response_str = json.dumps(raw_response_obj, ensure_ascii=False, indent=2)

    # Add selected names to metrics for the caller to save in Story.selected_names
    if lowest_set and isinstance(lowest_set, dict):
        sel_names_dict = lowest_set.get("names", lowest_set.get("Names"))
        if sel_names_dict and isinstance(sel_names_dict, dict):
            combined_metrics["selected_names"] = json.dumps(sel_names_dict, ensure_ascii=False)

    # Save to intermediate calls DB if configured
    if intermediate_db_path and intermediate_meta:
        try:
            save_intermediate_call(
                db_path=intermediate_db_path,
                image_list_id=intermediate_meta.get("image_list_id"),
                model_origin=final_model_name,
                prompt_modification=intermediate_meta.get("prompt_modification"),
                language=language,
                phase1_story=p1_story,
                phase1_story_clean=p1_story_clean,
                placeholders=placeholders,
                phase2_raw=p2_story,
                phase2_parsed=name_sets,
                selected_set=lowest_set,
                selected_index=selected_index,
                resolved_story=resolved_story,
                names_match=names_match,
            )
        except Exception as e:
            print(f"   [Intermediate DB] Warning: failed to save call: {e}")

    return resolved_story, raw_response_str, names_match, final_model_name, combined_metrics


def generate_speculative_story(
    model_config: dict,
    model_type: str,
    tags: list,
    language: str,
    localized_names: list,
    image_data: str = None,
    mime_type: str = "image/png",
    script_dir: Path = None,
    tracker: SlopTracker = None,
    lambda_value: float = 10.0,
    hard_mode: bool = False,
    hard_threshold: float = 0.01,
    intermediate_db_path=None,
    intermediate_meta=None,
):
    """
    Speculative naming generator with dynamic slop tracking.

    Implements the draft-verify-reject paradigm at the name level:
      Phase 1-2: reuse verbalized_sampling (5 name candidates with probs)
      Phase 3-4: verify against SlopTracker (baseline + runtime), select
                 by acceptance rule (exp(-λ×slop) weighted sampling or
                 hard threshold). Selected names are recorded in the tracker.
      Phase 6:   substitute placeholders (reused from multiphase_prob)

    This is a thin wrapper around generate_multiphase_prob_story that injects
    a speculative selection function. Phases 1, 2, and 6 are identical to
    verbalized sampling; only the SELECTION criterion changes.

    Returns: (story, raw_response, names_match, model_name, metrics)
    """
    if script_dir is None:
        script_dir = Path(__file__).resolve().parent.parent

    if tracker is None:
        print(f"   [Speculative] Warning: no tracker provided. "
              f"Falling back to verbalized sampling (λ has no effect).")

    def speculative_selection(name_sets):
        return select_speculative_set(
            name_sets,
            tracker=tracker,
            lambda_value=lambda_value,
            hard_mode=hard_mode,
            hard_threshold=hard_threshold,
        )

    return generate_multiphase_prob_story(
        model_config=model_config,
        model_type=model_type,
        tags=tags,
        language=language,
        localized_names=localized_names,
        image_data=image_data,
        mime_type=mime_type,
        script_dir=script_dir,
        selection_fn=speculative_selection,
        intermediate_db_path=intermediate_db_path,
        intermediate_meta=intermediate_meta,
    )

