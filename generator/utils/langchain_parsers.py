import re
import json
from typing import Any, Optional, List, Union
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.outputs import Generation, ChatGeneration

class ThinkStrippingJSONParser(JsonOutputParser):
    """
    A JSON output parser that strips <think>...</think> and <thought>...</thought> tags
    before parsing the JSON content. Useful for models like DeepSeek-R1.
    """
    
    def parse(self, text: str) -> Any:
        """
        Strip thinking tags and then parse the JSON.
        """
        if not isinstance(text, str):
            if hasattr(text, 'content'):
                text = text.content
            else:
                text = str(text)

        # 1. First attempt: Standard extraction of JSON block
        # This is incredibly robust as it ignores everything before { and after }
        match = re.search(r"(\{[\s\S]*\})", text)
        if match:
            potential_json = match.group(1)
            # Remove any thinking tags that might have leaked into the JSON block (rare)
            cleaned_json = re.sub(r"<(?:think|thought)>.*?</(?:think|thought)>", "", potential_json, flags=re.DOTALL | re.IGNORECASE)
            # Also handle unclosed tags at the very beginning of the block
            cleaned_json = re.sub(r"^<(?:think|thought)>[\s\S]*?(?=\{)", "", cleaned_json, flags=re.IGNORECASE)
            
            try:
                # Use json_repair for maximum robustness inside the extracted block
                try:
                    from json_repair import repair_json
                    repaired = repair_json(cleaned_json)
                    return json.loads(repaired)
                except ImportError: # json_repair not installed
                    # Fallback to standard json if repair fails or is missing
                    return json.loads(cleaned_json)
                except Exception: # Other errors from json_repair or json.loads(repaired)
                    # If repair failed, try standard json.loads on the cleaned_json
                    return json.loads(cleaned_json)
            except Exception:
                # If extraction failed to provide valid JSON, fall through to cleaning logic
                pass

        # 2. Backwards compatibility/Fallback: Clean the whole text and use base parser
        cleaned_text = re.sub(r"<(?:think|thought)>.*?</(?:think|thought)>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Handle unclosed tags at the start (don't use $ here to avoid over-stripping)
        cleaned_text = re.sub(r"<(?:think|thought)>", "", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.strip()
        
        return super().parse(cleaned_text)
