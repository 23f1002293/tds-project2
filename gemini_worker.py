import os
import json
import sys
import re
import time
import base64
import logging
import traceback
import asyncio

import google.generativeai as genai
import google.api_core.exceptions

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import httpx
import mimetypes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.flush = sys.stderr.flush


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(
        (genai.types.BlockedPromptException, google.api_core.exceptions.GoogleAPICallError)
    ),
)
def _generate_content_with_retry(
    model,
    prompt,
    task_name: str = "Gemini API call",
):
    start_time = time.time()
    logging.info(f"Initiating {task_name}...")
    response = model.generate_content(prompt)
    logging.info(f"{task_name} completed in {time.time() - start_time:.2f} seconds.")
    return response


def get_gemini_model(model_name: str = None):
    """Initializes and returns the Gemini model. Optionally, specify a model name."""
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        logging.error("GEMINI_API_KEY environment variable is not set.")
        raise ValueError("GEMINI_API_KEY is not set.")
    else:
        logging.info("GEMINI_API_KEY is set.")

    genai.configure(api_key=gemini_api_key)

    if model_name:
        logging.info(f"Attempting to use specified Gemini model: {model_name}")
        try:
            model_info = genai.get_model(model_name)
            if "generateContent" in model_info.supported_generation_methods:
                logging.info(f"Initializing Gemini model: {model_name}")
                return genai.GenerativeModel(model_name)
            else:
                logging.warning(
                    f"Model {model_name} does not support generateContent. "
                    "Falling back to default selection."
                )
        except Exception as e:
            logging.warning(
                f"Could not get model {model_name}. Falling back to default selection. "
                f"Error: {e}"
            )

    logging.info(
        "No specific model_name provided or specified model not suitable. "
        "Finding latest gemini-pro."
    )
    model_to_use = None
    latest_version = -1.0
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            if "preview" in model.name:
                continue
            match = re.search(r"gemini-(\d+\.?\d*)-pro", model.name)
            if match:
                version = float(match.group(1))
                if version > latest_version:
                    latest_version = version
                    model_to_use = model.name

    if not model_to_use:
        logging.error("Could not find a suitable Gemini pro model.")
        raise RuntimeError("Could not find a suitable Gemini pro model.")

    logging.info(f"Selected Gemini model: {model_to_use}")
    logging.info(f"Initializing Gemini model: {model_to_use}")

    try:
        return genai.GenerativeModel(model_to_use)
    except Exception as e:
        logging.error(f"Failed to instantiate Gemini model {model_to_use}: {e}")
        return None


async def generate_plan(
    quiz_content: str,
    initial_payload: dict,
    media_urls: list | None = None,
):
    logging.info("Entering generate_plan function.")

    model = get_gemini_model()
    if model is None:
        logging.error(
            "Gemini model initialization failed: "
            "get_gemini_model returned None for generate_plan."
        )
        raise RuntimeError("Gemini model initialization failed.")

    additional_context = ""

    if media_urls:
        media_processing_start_time = time.time()
        logging.info(f"Processing {len(media_urls)} media URLs concurrently.")

        async def process_single_media(media_url: str):
            try:
                logging.info(f"Fetching media from: {media_url}")
                async with httpx.AsyncClient() as client:
                    response = await client.get(media_url)
                    response.raise_for_status()
                    media_content = response.content
                    mime_type, _ = mimetypes.guess_type(media_url)
                    if not mime_type:
                        mime_type = "application/octet-stream"
                    logging.info(
                        "Successfully fetched media "
                        f"({len(media_content)} bytes, type: {mime_type}) "
                        f"from {media_url}"
                    )

                    multimodal_model = get_gemini_model(model_name="gemini-1.5-pro")
                    if multimodal_model is None:
                        logging.warning(
                            "Multimodal model (gemini-1.5-pro) not available, "
                            "skipping media analysis for this URL."
                        )
                        return None

                    media_part = {"mime_type": mime_type, "data": media_content}
                    multimodal_prompt = [
                        media_part,
                        (
                            "Analyze this media for any clues, secrets, or important "
                            "information related to a multi-step task or quiz. "
                            "Extract any text, codes, or instructions. "
                            "Respond concisely with only the extracted clue."
                        ),
                    ]
                    logging.info(
                        "Sending media to multimodal LLM. Prompt (text part "
                        f"truncated to 100 chars): {multimodal_prompt[1][:100]}"
                    )
                    multimodal_response = _generate_content_with_retry(
                        multimodal_model, multimodal_prompt, "Multimodal analysis"
                    )
                    clue = multimodal_response.text.strip()
                    if clue:
                        logging.info(f"Extracted clue from media: {clue[:100]}")
                        return f"\n\nClue from {media_url}: {clue}"
                    logging.info(f"No clue extracted from media: {media_url}")
                    return None
            except Exception as e:
                logging.error(f"Error processing media {media_url}: {e}")
                return None

        processing_tasks = [process_single_media(url) for url in media_urls]
        clues = await asyncio.gather(*processing_tasks)
        additional_context = "".join(c for c in clues if c is not None)
        logging.info(
            "Media processing completed in "
            f"{time.time() - media_processing_start_time:.2f} seconds."
        )

    initial_payload_str = json.dumps(initial_payload, indent=2)

    planning_prompt = f"""
You are an intelligent agent tasked with solving a multi-step challenge presented on a webpage. Your goal is to analyze the provided HTML content and create a JSON plan to accomplish the main objective.

Instructions:

1. Analyze the entire context:
   - Read all visible text.
   - Examine all links (<a> tags) and embedded media (<audio>, <video>).
   - The solution may require combining information from:
     - the HTML text,
     - a linked file (CSV, JSON, PDF, etc.),
     - or transcribed media.

2. Determine the primary action:
   - Decide the single most important action needed to solve the task.
   - Typically: fetch a URL (file or HTML) and then process its contents.

3. Code execution (STRICTLY for computation / data manipulation):

   WHEN YOU NEED PYTHON:
   - Set "processing_task" to a string that contains the marker "execute_python_code:".
   - Immediately after that marker, include a fenced code block:

     ```
     # your code here
     ```

   - The code MUST:
     - Read from the provided `data_content` (string with fetched data) or `html_content` (string with current page HTML).
     - Perform all necessary computations (e.g., filtering rows, aggregating numbers, combining clues).
     - Print ONLY the final answer to stdout.

   WHEN YOU DO NOT NEED PYTHON:
   - Do NOT mention "execute_python_code" anywhere.
   - In that case, "processing_task" must be a short natural-language description,
     like:
       "transcribe the audio and extract the secret code"
       "read the CSV and describe the pattern in one sentence"
       "extract the password from the HTML text"

4. Plan schema (must always be followed):

   Respond ONLY with a JSON object with a single key "plan".
   The value of "plan" MUST be a JSON object with these fields:

   - "task_type": must be "fetch_and_submit".
   - "fetch_url": the single most relevant URL to fetch next (or null if no fetch is needed).
   - "processing_task":
       - EITHER includes "execute_python_code:" plus a `````` fenced block,
       - OR is a simple natural-language description with no code.
   - "submit_url": the full URL where the final answer will be POSTed.
   - "payload": the JSON payload for the submission. Wherever the final answer
     should go, you MUST use the placeholder string "__ANSWER__" instead
     of guessing the value yourself.

Archetypal examples (for style, not content):

- Example A (text-only extraction):
  - If the answer is clearly in the HTML text, do NOT use execute_python_code.
  - Use a natural-language "processing_task" like
    "find the secret code in the HTML and return it exactly".

- Example B (CSV computation):
  - If a CSV link contains numbers that must be aggregated or filtered
    (e.g., "sum the values in the second column where the first column
    is greater than a cutoff"), then:
      - "fetch_url" should point to the CSV.
      - "processing_task" MUST use execute_python_code and a `````` block
        that:
          - reads `data_content` as CSV,
          - performs the computation,
          - prints the numeric answer.

Your context:

Additional media-derived clues (if any):
---
{additional_context}
---

Initial JSON Payload (your context):
---
{initial_payload_str}
---

Webpage Content (HTML instructions):
---
{quiz_content}
---
""".strip()

    logging.info(
        "Prompt for generate_plan (truncated to 500 chars): "
        f"{planning_prompt[:500]}"
    )

    try:
        response = _generate_content_with_retry(
            model, planning_prompt, "Plan generation"
        )
        logging.info(
            "Received response from Gemini API (generate_plan): "
            f"{response.text[:500]}"
        )
    except (
        genai.types.BlockedPromptException,
        google.api_core.exceptions.GoogleAPICallError,
    ) as e:
        logging.warning(
            f"Primary model failed ({e}). Attempting fallback to gemini-1.5-pro."
        )
        fallback_model = get_gemini_model(model_name="gemini-1.5-pro")
        response = _generate_content_with_retry(
            fallback_model, planning_prompt, "Plan generation (fallback model)"
        )
        logging.info(
            "Received response from fallback Gemini API (generate_plan): "
            f"{response.text[:500]}"
        )

    logging.info("Exiting generate_plan function.")
    return json.dumps({"llm_response": response.text})


def extract_answer(content: str, question: str):
    logging.info("Entering extract_answer function.")

    model = get_gemini_model()
    if model is None:
        logging.error(
            "Gemini model initialization failed: "
            "get_gemini_model returned None for extract_answer."
        )
        raise RuntimeError("Gemini model initialization failed.")

    prompt = f"""
You are an expert at finding specific information. Analyze the following content and answer the question.

Respond with ONLY the answer, and nothing else. Do not add any formatting or explanations.

Content: ---
{content}
---

Question: ---
{question}
---
""".strip()

    logging.info(f"Prompt for extract_answer (truncated to 500 chars): {prompt[:500]}")
    try:
        response = _generate_content_with_retry(
            model, prompt, "Answer extraction"
        )
        logging.info(
            "Received response from Gemini API (extract_answer): "
            f"{response.text[:500]}"
        )
    except (genai.types.BlockedPromptException, google.api_core.exceptions.GoogleAPICallError) as e:
        logging.warning(
            f"Primary model failed ({e}). Attempting fallback to gemini-1.5-pro."
        )
        model = get_gemini_model(model_name="gemini-1.5-pro")
        response = _generate_content_with_retry(
            model, prompt, "Answer extraction (fallback model)"
        )
        logging.info(
            "Received response from fallback Gemini API (extract_answer): "
            f"{response.text[:500]}"
        )

    logging.info("Exiting extract_answer function.")
    return json.dumps({"answer": response.text.strip()})


def process_data(data_content: str, processing_task: str):
    logging.info("Entering process_data function.")

    model = get_gemini_model()
    if model is None:
        logging.error(
            "Gemini model initialization failed: "
            "get_gemini_model returned None for process_data."
        )
        raise RuntimeError("Gemini model initialization failed.")

    prompt = f"""
You are a data processing specialist. Perform the following task on the given data.

Respond with ONLY the final result, and nothing else. Do not add any formatting, explanations, or quotes.

Data: ---
{data_content}
---

Task: ---
{processing_task}
---
""".strip()

    logging.info(f"Prompt for process_data (truncated to 500 chars): {prompt[:500]}")
    try:
        response = _generate_content_with_retry(
            model, prompt, "Data processing"
        )
        logging.info(
            "Received response from Gemini API (process_data): "
            f"{response.text[:500]}"
        )
    except (genai.types.BlockedPromptException, google.api_core.exceptions.GoogleAPICallError) as e:
        logging.warning(
            f"Primary model failed ({e}). Attempting fallback to gemini-1.5-pro."
        )
        model = get_gemini_model(model_name="gemini-1.5-pro")
        response = _generate_content_with_retry(
            model, prompt, "Data processing (fallback model)"
        )
        logging.info(
            "Received response from fallback Gemini API (process_data): "
            f"{response.text[:500]}"
        )

    logging.info("Exiting process_data function.")
    return json.dumps({"processed_data": response.text.strip()})


def process_audio_with_gemini(audio_content: bytes, processing_task: str):
    logging.info("Entering process_audio_with_gemini function.")

    model = get_gemini_model()
    if model is None:
        logging.error(
            "Gemini model initialization failed: "
            "get_gemini_model returned None for process_audio_with_gemini."
        )
        raise RuntimeError("Gemini model initialization failed.")

    audio_part = {
        "mime_type": "audio/ogg",
        "data": audio_content,
    }

    prompt = [
        audio_part,
        (
            "Analyze the provided audio and "
            f"{processing_task}. Respond with ONLY the final answer, and nothing else."
        ),
    ]

    logging.info(
        "Prompt for process_audio_with_gemini (text part truncated to 500 chars): "
        f"{prompt[1][:500]}"
    )
    logging.info(
        f"Audio content size for process_audio_with_gemini: {len(audio_content)} bytes."
    )

    try:
        response = _generate_content_with_retry(
            model, prompt, "Audio processing"
        )
        logging.info(
            "Received response from Gemini API (process_audio_with_gemini): "
            f"{response.text[:500]}"
        )
    except (genai.types.BlockedPromptException, google.api_core.exceptions.GoogleAPICallError) as e:
        logging.warning(
            f"Primary model failed ({e}). Attempting fallback to gemini-1.5-pro."
        )
        model = get_gemini_model(model_name="gemini-1.5-pro")
        response = _generate_content_with_retry(
            model, prompt, "Audio processing (fallback model)"
        )
        logging.info(
            "Received response from fallback Gemini API (process_audio_with_gemini): "
            f"{response.text[:500]}"
        )

    logging.info("Exiting process_audio_with_gemini function.")
    return json.dumps({"processed_data": response.text.strip()})


if __name__ == "__main__":

    async def main_async():
        input_data = {}
        try:
            input_data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON input to worker: {e}"}))
            sys.stdout.flush()
            sys.exit(1)
        except Exception as e:
            print(json.dumps({"error": f"Error loading input to worker: {e}"}))
            sys.stdout.flush()
            sys.exit(1)

        try:
            task = input_data.get("task")

            if task == "generate_plan":
                result_json = await generate_plan(
                    input_data["quiz_content"],
                    input_data["initial_payload"],
                    input_data.get("media_urls"),
                )
                print(result_json)
                sys.stdout.flush()

            elif task == "extract_answer":
                result_json = extract_answer(
                    input_data["content"], input_data["question"]
                )
                print(result_json)
                sys.stdout.flush()

            elif task == "process_data":
                result_json = process_data(
                    input_data["data_content"], input_data["processing_task"]
                )
                print(result_json)
                sys.stdout.flush()

            elif task == "process_audio_with_gemini":
                audio_content_base64 = input_data["audio_content"]
                audio_bytes = base64.b64decode(audio_content_base64)
                result_json = process_audio_with_gemini(
                    audio_bytes, input_data["processing_task"]
                )
                print(result_json)
                sys.stdout.flush()

            else:
                print(json.dumps({"error": f"Unknown task: {task}"}))
                sys.stdout.flush()
                sys.exit(1)

        except Exception as e:
            error_details = {"error": str(e), "traceback": traceback.format_exc()}
            print(json.dumps(error_details))
            sys.stdout.flush()
            sys.exit(1)

    asyncio.run(main_async())
