import time
import traceback
import threading
import json
import os
from contextlib import contextmanager
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from openai.lib._parsing._completions import LengthFinishReasonError
from utils.langchain_parsers import ThinkStrippingJSONParser
from utils.robust_parse import parse_with_repair
def is_langfuse_enabled() -> bool:
    """Check if Langfuse tracing is enabled."""
    if os.getenv("LANGFUSE_ENABLED", "true").lower() == "false":
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

# Initialize Langfuse CallbackHandler and client if credentials are set and enabled
_langfuse_handler = None
_langfuse_client = None

if is_langfuse_enabled():
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
        _langfuse_handler = CallbackHandler()
        _langfuse_client = get_client()
    except Exception as e:
        print(f"Warning: Failed to initialize Langfuse: {e}")

def _get_langfuse_handler():
    global _langfuse_handler
    if not _langfuse_handler and is_langfuse_enabled():
        try:
            from langfuse.langchain import CallbackHandler
            _langfuse_handler = CallbackHandler()
        except Exception as e:
            print(f"Warning: Failed to initialize Langfuse CallbackHandler: {e}")
    return _langfuse_handler

def get_langfuse_client():
    global _langfuse_client
    if is_langfuse_enabled():
        return _langfuse_client
    return None

def observe_custom(name, get_input_fn=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if is_langfuse_enabled() and _langfuse_client:
                input_data = None
                try:
                    if get_input_fn:
                        input_data = get_input_fn(*args, **kwargs)
                except Exception as e:
                    print(f"Error extracting Langfuse inputs: {e}")
                
                with _langfuse_client.start_as_current_observation(
                    name=name,
                    input=input_data
                ) as span:
                    result = func(*args, **kwargs)
                    
                    # Determine output to log
                    output_val = result
                    if isinstance(result, tuple):
                        if len(result) > 0:
                            output_val = result[0]
                        else:
                            output_val = None
                            
                    if isinstance(result, dict):
                        if "parsed" in result:
                            if result["parsed"] is not None:
                                try:
                                    output_val = result["parsed"].model_dump() if hasattr(result["parsed"], "model_dump") else str(result["parsed"])
                                except:
                                    output_val = str(result["parsed"])
                            else:
                                output_val = f"<Error: {result.get('error', 'Unknown Error')}>"
                                
                    if output_val is not None:
                        # Mask base64 or long image urls/data
                        if isinstance(output_val, str) and len(output_val) > 2000 and not output_val.startswith("http"):
                            span.update(output=f"<Base64 Image/Data, length: {len(output_val)}>")
                        else:
                            span.update(output=output_val)
                    else:
                        span.update(output="<None/Failed>")
                        
                    return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Global thread-safe cache for ChatOpenAI client instances
_model_cache = {}
_model_cache_lock = threading.Lock()

def load_model(host, model, apikey, temperature=0.7, json_schema = None, think = None, timeout = 300, min_p = None):
    global _model_cache
    
    # Create a unique key for caching the model client instance
    schema_str = json.dumps(json_schema, sort_keys=True) if json_schema else ""
    cache_key = (host, model, apikey, temperature, think, timeout, min_p, schema_str)
    
    with _model_cache_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]

    # Base configuration
    config = {
        "base_url": host,
        "model": model,
        "api_key": apikey,
        "temperature": temperature,
        "top_p": 0.9,  # Fixed across all models for consistent sampling
        "max_tokens": 5000,
        "timeout": timeout, # Correct parameter name for modern ChatOpenAI
    }

    if think is not None:
        if "extra_body" not in config:
            config["extra_body"] = {}
        
        if isinstance(think, bool):
            config["extra_body"]["think"] = think
            config["extra_body"]["reasoning_effort"] = "none" if not think else "high"
        elif isinstance(think, str):
            config["extra_body"]["reasoning_effort"] = think

    if min_p is not None:
        if "extra_body" not in config:
            config["extra_body"] = {}
        config["extra_body"]["min_p"] = min_p

    llm = ChatOpenAI(**config)
    if json_schema:
        # Set the model to use structured output with the provided JSON schema
        llm = llm.with_structured_output(json_schema, method="json_schema")
        
    with _model_cache_lock:
        _model_cache[cache_key] = llm
        
    return llm

def create_structured_chain(llm, json_schema):
    """
    Creates a chain that uses ThinkStrippingJSONParser for robust structured output.
    """
    parser = ThinkStrippingJSONParser(pydantic_object=None) # We use dict output by default
    # Note: pydantic_object can be passed to parser if desired
    return llm | parser

def _extract_tokens(response):
    """
    Extract input and output tokens from a LangChain response message.
    """
    input_tokens = None
    output_tokens = None
    
    # 1. Try standardized usage_metadata
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        print(f"DEBUG: usage_metadata: {response.usage_metadata}")
        um = response.usage_metadata
        if isinstance(um, dict):
            input_tokens = um.get('input_tokens')
            output_tokens = um.get('output_tokens')
        else:
            input_tokens = getattr(um, 'input_tokens', None)
            output_tokens = getattr(um, 'output_tokens', None)
            
    # 2. Try response_metadata (provider-specific)
    if (input_tokens is None or output_tokens is None) and hasattr(response, 'response_metadata'):
        rm = response.response_metadata
        
        # OpenAI / OpenRouter
        if 'token_usage' in rm:
            input_tokens = input_tokens or rm['token_usage'].get('prompt_tokens')
            output_tokens = output_tokens or rm['token_usage'].get('completion_tokens')
        
        # Ollama
        elif 'prompt_eval_count' in rm:
            input_tokens = input_tokens or rm.get('prompt_eval_count')
            output_tokens = output_tokens or rm.get('eval_count')
            
        # Google
        elif 'usage_metadata' in rm:
            input_tokens = input_tokens or rm['usage_metadata'].get('prompt_token_count')
            output_tokens = output_tokens or rm['usage_metadata'].get('candidates_token_count')

    return input_tokens, output_tokens

def _inject_min_p_metadata(model, config: dict) -> dict:
    if config is None:
        handler = _get_langfuse_handler()
        callbacks = [handler] if handler else []
        config = {"callbacks": callbacks}
    else:
        config = dict(config)
        
    model_instance = model
    from langchain_core.runnables import RunnableSequence
    if isinstance(model, RunnableSequence):
        model_instance = model.first
        
    if hasattr(model_instance, "extra_body") and isinstance(model_instance.extra_body, dict):
        min_p_val = model_instance.extra_body.get("min_p")
        if min_p_val is not None:
            if "metadata" not in config:
                config["metadata"] = {}
            else:
                config["metadata"] = dict(config["metadata"])
                
            config["metadata"]["min_p"] = min_p_val
            
            # Add to langfuse_tags
            if "langfuse_tags" not in config["metadata"]:
                config["metadata"]["langfuse_tags"] = []
            elif isinstance(config["metadata"]["langfuse_tags"], str):
                config["metadata"]["langfuse_tags"] = [config["metadata"]["langfuse_tags"]]
            else:
                config["metadata"]["langfuse_tags"] = list(config["metadata"]["langfuse_tags"])
                
            tag_str = f"min_p={min_p_val}"
            if tag_str not in config["metadata"]["langfuse_tags"]:
                config["metadata"]["langfuse_tags"].append(tag_str)
                
    return config

#@observe_custom("generate_response", lambda prompt, system_prompt, *args, **kwargs: {"prompt": prompt, "system_prompt": system_prompt})
def generate_response(prompt, system_prompt, model, supports_system_instructions = True, config = None):
    from langchain_core.messages import BaseMessage
    if isinstance(prompt, list) and all(isinstance(m, BaseMessage) for m in prompt):
        messages_formatted = prompt
    else:
        if not supports_system_instructions:
            # Merge system prompt into human message for models that don't support system instructions
            prompt_template = ChatPromptTemplate.from_messages([
                HumanMessage(content=f"{system_prompt}\n\n{prompt}")
            ])
        else:
            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
        messages_formatted = prompt_template.format_messages()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            # Check if model is a chain (RunnableSequence)
            from langchain_core.runnables import RunnableSequence
            
            # Retrieve the CallbackHandler dynamically
            config = _inject_min_p_metadata(model, config)
            
            if isinstance(model, RunnableSequence):
                # Step 1: LLM result
                llm_response = model.first.invoke(messages_formatted, config=config)
                
                # Step 2: Metrics
                input_tokens, output_tokens = _extract_tokens(llm_response)
                metrics = {
                    "execution_time": time.time() - start_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
                
                # Step 3: Result through the rest of the chain
                # Note: we skip model.first and run the rest
                final_result = llm_response.content
                # If there's a parser/transformer after the first step
                for step in model.steps[1:]:
                    final_result = step.invoke(final_result)
                
                return final_result, metrics
            else:
                # Standard model
                response = model.invoke(messages_formatted, config=config)
                input_tokens, output_tokens = _extract_tokens(response)
                metrics = {
                    "execution_time": time.time() - start_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
                
                content = response.content if hasattr(response, 'content') else response
                return content, metrics
                
        except LengthFinishReasonError as e:
            return None, {}
        except Exception as e:
            error_str = str(e).lower()
            is_transient = any(msg in error_str for msg in ["429", "rate limit", "timeout", "connection", "connect", "overloaded", "bad gateway", "service unavailable"])
            
            if is_transient:
                wait_time = 60 if ("429" in error_str or "rate limit" in error_str) else 15
                err_type = "Rate limit (429)" if ("429" in error_str or "rate limit" in error_str) else "Transient error (timeout/connection/overload)"
                print(f"\n[!] {err_type} detected. Waiting {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
                
            print(f"Error generating response: {e}")
            try:
                import datetime
                import os
                
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                model_name = getattr(model, "model_name", "unknown_model")
                
                log_entry = f"[{timestamp}] [{model_name}] {str(e)}\n"
                
                existing_content = ""
                if os.path.exists("gen_error.txt"):
                    with open("gen_error.txt", "r", encoding="utf-8") as f:
                        existing_content = f.read()
                
                with open("gen_error.txt", "w", encoding="utf-8") as f:
                    f.write(log_entry + existing_content)
                    
            except Exception as log_err:
                print(f"Failed to log error to file: {log_err}")
                
            return None, {}
    return None, {}

#@observe_custom("generate_multimodal_response", lambda prompt_text, system_prompt, model, image_data, mime_type="image/jpeg", *args, **kwargs: {"prompt_text": prompt_text, "system_prompt": system_prompt, "mime_type": mime_type, "has_image": image_data is not None})
def generate_multimodal_response(prompt_text, system_prompt, model, image_data, mime_type="image/jpeg", supports_system_instructions = True, config = None):
    """
    Generate a response using a multimodal model with text and image input.
    image_data: base64 encoded image string
    mime_type: mime type of the image (default: image/jpeg)
    """
    from langchain_core.messages import BaseMessage
    if isinstance(prompt_text, list) and all(isinstance(m, BaseMessage) for m in prompt_text):
        messages_formatted = prompt_text
    else:
        if not supports_system_instructions:
             messages = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": f"{system_prompt}\n\n{prompt_text}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                        },
                    ]
                )
             ]
        else:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                        },
                    ]
                )
            ]
        messages_formatted = messages

    max_retries = 3
    for attempt in range(max_retries):
        try:
            import time
            start_time = time.time()
            
            # Check if model is a chain
            from langchain_core.runnables import RunnableSequence
            
            # Retrieve the CallbackHandler dynamically
            config = _inject_min_p_metadata(model, config)
            
            if isinstance(model, RunnableSequence):
                # Step 1: LLM result
                llm_response = model.first.invoke(messages_formatted, config=config)
                
                # Step 2: Metrics
                input_tokens, output_tokens = _extract_tokens(llm_response)
                metrics = {
                    "execution_time": time.time() - start_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
                
                # Step 3: Result through rest of chain
                final_result = llm_response.content
                for step in model.steps[1:]:
                    final_result = step.invoke(final_result)
                
                return final_result, metrics
            else:
                # Standard model
                response = model.invoke(messages_formatted, config=config)
                input_tokens, output_tokens = _extract_tokens(response)
                metrics = {
                    "execution_time": time.time() - start_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
                
                content = response.content if hasattr(response, 'content') else response
                return content, metrics
        except Exception as e:
            error_str = str(e).lower()
            is_transient = any(msg in error_str for msg in ["429", "rate limit", "timeout", "connection", "connect", "overloaded", "bad gateway", "service unavailable"])
            
            if is_transient:
                wait_time = 60 if ("429" in error_str or "rate limit" in error_str) else 15
                err_type = "Rate limit (429)" if ("429" in error_str or "rate limit" in error_str) else "Transient error (timeout/connection/overload)"
                print(f"\n[!] {err_type} detected during multimodal call. Waiting {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue

            print(f"Error generating multimodal response: {e}")
            # Log error similar to generate_response
            try:
                import datetime
                import os
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                model_name = getattr(model, "model_name", "unknown_model")
                log_entry = f"[{timestamp}] [MULTIMODAL] [{model_name}] {str(e)}\n"
                existing_content = ""
                if os.path.exists("gen_error.txt"):
                    with open("gen_error.txt", "r", encoding="utf-8") as f:
                        existing_content = f.read()
                with open("gen_error.txt", "w", encoding="utf-8") as f:
                    f.write(log_entry + existing_content)
            except:
                pass
            return None, {}
    return None, {}

#@observe_custom("generate_image_response", lambda prompt, model, size="1024x1024", quality="standard", n=1, *args, **kwargs: {"prompt": prompt, "size": size, "quality": quality, "n": n})
def generate_image_response(prompt, model, size="1024x1024", quality="standard", n=1):
    """
    Generate an image using OpenRouter's specific chat-completion-based image generation.
    Returns the base64 encoded image string.
    """
    try:
        import time
        start_time = time.time()
        import httpx
        import json
        
        # Extract configuration from the ChatOpenAI model instance
        api_key = model.openai_api_key.get_secret_value()
        base_url = model.openai_api_base
        model_name = model.model_name
        
        # Ensure base_url ends correctly for appending path if needed, 
        # but ChatOpenAI usually has ".../v1".
        # OpenRouter example: https://openrouter.ai/api/v1/chat/completions
        # if base_url is ".../v1", we append "/chat/completions"
 
        data = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "modalities": ["image"]
        }
        
        target_url = f"{base_url}/chat/completions"

        if base_url.startswith("https://api.openai.com"):
            # remove messages and modalities
            del data['messages']
            del data['modalities']
            data['prompt'] = prompt
            data['size'] = "1024x1536"
            data['quality'] = "medium"
            data['n'] = 1
            target_url = base_url

        # Handle potential double slashes if base_url ends in /
        if base_url.endswith("/"):
            # remove the last slash
            target_url = base_url[:-1]
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Make the request directly using httpx Client (sync)
        with httpx.Client() as client:
            response = client.post(
                url=target_url,
                headers=headers,
                json=data,
                timeout=60 # Image gen might take time
            )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract image from OpenRouter response format
        # Structure: choices[0].message.images[0].image_url.url (base64 data uri)
        if result.get("choices"):
            message = result["choices"][0]["message"]
            # Some models might put it in content, but OpenRouter specifically mentioned 'images' list in message
            # for flux/etc.
            if message.get("images"):
                for image in message["images"]:
                     # It's a data URL: "data:image/png;base64,....."
                     image_url = image.get("image_url", {}).get("url")
                     if image_url and image_url.startswith("data:"):
                         # Extract plain base64
                         content = image_url.split(",", 1)[1]
                         metrics = {
                             "execution_time": time.time() - start_time,
                             "input_tokens": result.get("usage", {}).get("prompt_tokens"),
                             "output_tokens": result.get("usage", {}).get("completion_tokens")
                         }
                         return content, metrics
                     elif image_url:
                         metrics = {
                             "execution_time": time.time() - start_time,
                             "input_tokens": result.get("usage", {}).get("prompt_tokens"),
                             "output_tokens": result.get("usage", {}).get("completion_tokens")
                         }
                         return image_url, metrics

        if base_url.startswith("https://api.openai.com"):
            print("OpenAI image recibida")
            # save raw json to analyze
            with open("openai_image.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            if result.get("data"):
                for image in result["data"]:
                    # estructura: "data": [ { "b64_json": ... } ]
                    image_b64 = image.get("b64_json")
                    if image_b64:
                        # Extract plain base64
                        content = image_b64
                        metrics = {
                            "execution_time": time.time() - start_time,
                            "input_tokens": result.get("usage", {}).get("input_tokens"),
                            "output_tokens": result.get("usage", {}).get("output_tokens")
                        }
                        return content, metrics
                    elif image_url:
                        metrics = {
                            "execution_time": time.time() - start_time,
                            "input_tokens": result.get("usage", {}).get("prompt_tokens"),
                            "output_tokens": result.get("usage", {}).get("completion_tokens")
                        }
                        return image_url, metrics
            
        return None, {}
        
    except Exception as e:
        print(f"Error generating image response: {e}")
        try:
            import datetime
            import os
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_name = getattr(model, "model_name", "unknown_model")
            log_entry = f"[{timestamp}] [IMAGE] [{model_name}] {str(e)}\n"
            existing_content = ""
            if os.path.exists("gen_error.txt"):
                with open("gen_error.txt", "r", encoding="utf-8") as f:
                    existing_content = f.read()
            with open("gen_error.txt", "w", encoding="utf-8") as f:
                f.write(log_entry + existing_content)
        except:
            pass
        return None, {}

def _extract_metadata(response, duration):
    metadata = {
        "generation_time": duration,
        "input_tokens": 0,
        "output_tokens": 0
    }
    
    if not response:
        return metadata

    if hasattr(response, 'response_metadata'):
        usage = response.response_metadata.get('token_usage', {})
        metadata["input_tokens"] = usage.get('prompt_tokens', 0)
        metadata["output_tokens"] = usage.get('completion_tokens', 0)
    
    # Some providers use usage_metadata
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        metadata["input_tokens"] = response.usage_metadata.get('input_tokens', metadata["input_tokens"])
        metadata["output_tokens"] = response.usage_metadata.get('output_tokens', metadata["output_tokens"])
    
    return metadata

#@observe_custom("generate_structured_response", lambda prompt, system_prompt, model, pydantic_schema, *args, **kwargs: {"prompt": prompt, "system_prompt": system_prompt, "schema_name": getattr(pydantic_schema, "__name__", str(pydantic_schema))})
def generate_structured_response(prompt, system_prompt, model, pydantic_schema, method="json_schema", config = None):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]

    # Use include_raw=True to get the message object (and metadata) along with the parsed object.
    # This allows us to recover from parsing errors without a second expensive LLM call.
    # method can be "json_schema" or "json_mode" (the latter uses json_object)
    structured_llm = model.with_structured_output(pydantic_schema, method=method, include_raw=True)
    
    max_retries = 3
    for attempt in range(max_retries):
        start_time = time.time()
        try:
            # Retrieve the CallbackHandler dynamically
            config = _inject_min_p_metadata(model, config)
            
            response_dict = structured_llm.invoke(messages, config=config)
            end_time = time.time()
            
            parsed = response_dict.get('parsed')
            raw = response_dict.get('raw')
            parsing_error = response_dict.get('parsing_error')
            
            if parsed and not parsing_error:
                # Success! Extract metadata from raw response
                metadata = _extract_metadata(raw, end_time - start_time)
                res = {
                    "parsed": parsed,
                    "metadata": metadata
                }
                return res
            
            # If we are here, either 'parsed' is None or 'parsing_error' is present.
            # But we HAVE the raw response in 'raw' (which is an AIMessage).
            if raw and hasattr(raw, 'content'):
                content = raw.content
                print(f"\n[LLM INFO] Structured parsing failed but raw content is available. Attempting robust recovery...")
                
                data, diag = parse_with_repair(content)
                if data:
                    try:
                        parsed = pydantic_schema.model_validate(data)
                        print(f"  [+] Success: Recovered JSON from initial structured response. Repair used: {diag['used_json_repair']}")
                        res = {
                            "parsed": parsed,
                            "metadata": _extract_metadata(raw, end_time - start_time),
                            "recovered": True,
                            "raw_response": content
                        }
                        return res
                    except Exception as ve:
                        print(f"  [!] Recovered JSON failed schema validation: {ve}")
                else:
                     print(f"  [!] Could not repair/extract JSON from initial raw content: {diag['error_message']}")

            # Total failure or no content returned
            error_info = str(parsing_error) if parsing_error else "Incomplete response"
            res = {
                "parsed": None,
                "error": error_info,
                "metadata": _extract_metadata(raw, end_time - start_time)
            }
            return res

        except Exception as e:
            error_str = str(e).lower()
            is_transient = any(msg in error_str for msg in ["429", "rate limit", "timeout", "connection", "connect", "overloaded", "bad gateway", "service unavailable"])
            
            if is_transient:
                wait_time = 60 if ("429" in error_str or "rate limit" in error_str) else 15
                err_type = "Rate limit (429)" if ("429" in error_str or "rate limit" in error_str) else "Transient error (timeout/connection/overload)"
                print(f"\n[!] {err_type} detected during structured call. Waiting {wait_time} seconds and retrying... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue

            # Fallback: Something went wrong with the API call itself (Refusal, Non-transient errors, etc.)
            print(f"\n[LLM ERROR] Exception during structured invoke: {e}. Attempting manual fallback call...")
            
            try:
                # Re-call as last resort if no raw content was returned in the first try
                raw_response = model.invoke(messages, config=config)
                content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                
                data, diag = parse_with_repair(content)
                if data:
                    parsed = pydantic_schema.model_validate(data)
                    print(f"  [+] Success: Parsed JSON using fallback call. Repair used: {diag['used_json_repair']}")
                    res = {
                        "parsed": parsed,
                        "metadata": _extract_metadata(raw_response, 0),
                        "recovered": True,
                        "raw_response": content
                    }
                    return res
                else:
                    print(f"  [!] Fallback call also failed to yield valid JSON: {diag['error_message']}")
            except Exception as fallback_e:
                fb_error_str = str(fallback_e).lower()
                is_fb_transient = any(msg in fb_error_str for msg in ["429", "rate limit", "timeout", "connection", "connect", "overloaded", "bad gateway", "service unavailable"])
                if is_fb_transient:
                    wait_time = 60 if ("429" in fb_error_str or "rate limit" in fb_error_str) else 15
                    err_type = "Rate limit (429)" if ("429" in fb_error_str or "rate limit" in fb_error_str) else "Transient error (timeout/connection/overload)"
                    print(f"  [!] {err_type} detected during fallback. Waiting {wait_time} seconds before next attempt...")
                    time.sleep(wait_time)
                    continue
                print(f"  [!] Fallback call failed: {fallback_e}")

            res = {
                "parsed": None,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "metadata": {"error": str(e)}
            }
            return res
    return None

def flush_langfuse():
    """Flush Langfuse traces if enabled."""
    if is_langfuse_enabled() and _langfuse_client:
        try:
            _langfuse_client.flush()
        except Exception as e:
            print(f"Failed to flush Langfuse traces: {e}")