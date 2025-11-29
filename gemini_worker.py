import os
import json
import sys
import re
import google.generativeai as genai
import base64
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions
import httpx # Import httpx for media downloading
import mimetypes # Import mimetypes for media type detection
import asyncio # Import asyncio for running async functions

# Configure logging for the worker to flush immediately
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)

# Ensure the StreamHandler flushes immediately
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.flush = sys.stderr.flush

# Define retry strategy for Gemini API calls
# We retry on specific Gemini API errors or blocked prompts (e.g., token limits or safety issues)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
       retry=retry_if_exception_type((genai.types.BlockedPromptException, google.api_core.exceptions.GoogleAPICallError)))
def _generate_content_with_retry(model, prompt):
    return model.generate_content(prompt)

def get_gemini_model(model_name: str = None):

    """Initializes and returns the Gemini model. Optionally, specify a model name."""

    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    if not gemini_api_key:
        logging.error("GEMINI_API_KEY environment variable is not set.")
        raise ValueError("GEMINI_API_KEY is not set.")
    else:
        logging.info("GEMINI_API_KEY is set.")

    genai.configure(api_key=gemini_api_key)

    model_to_use = None
    if model_name:
        logging.info(f"Attempting to use specified Gemini model: {model_name}")
        try:
            model_info = genai.get_model(model_name)
            if 'generateContent' in model_info.supported_generation_methods:
                model_to_use = model_name
            else:
                logging.warning(f"Model {model_name} does not support generateContent. Falling back to default selection.")
                model_name = None # Fallback to default selection below
        except Exception as e:
            logging.warning(f"Could not get model {model_name}. Falling back to default selection. Error: {e}")
            model_name = None

    if not model_to_use:
        logging.info("No specific model_name provided or specified model not suitable. Finding latest gemini-pro.")
        latest_version = -1.0
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                match = re.search(r'gemini-(\d+\.?\d*)-pro', model.name)
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

async def generate_plan(quiz_content: str, initial_payload: dict, media_urls: list = None):
    logging.info("Entering generate_plan function.")
    model = get_gemini_model()
    if model is None:
        logging.error("Gemini model initialization failed: get_gemini_model returned None for generate_plan.")
        raise RuntimeError("Gemini model initialization failed.")
    additional_context = ""
    if media_urls:
        logging.info(f"Processing {len(media_urls)} media URLs.")
        async with httpx.AsyncClient() as client:
            for media_url in media_urls:
                try:
                    logging.info(f"Fetching media from: {media_url}")
                    response = await client.get(media_url)
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    media_content = response.content
                    mime_type, _ = mimetypes.guess_type(media_url)
                    if not mime_type: # Fallback for unknown types
                        mime_type = 'application/octet-stream'
                    
                    logging.info(f"Successfully fetched media ({len(media_content)} bytes, type: {mime_type}) from {media_url}")
                    
                    # Send media to multimodal LLM for clue extraction
                    # Using gemini-1.5-pro for multimodal capabilities (audio/video)
                    multimodal_model = get_gemini_model(model_name="gemini-1.5-pro") 
                    if multimodal_model is None:
                        logging.warning("Multimodal model (gemini-1.5-pro) not available, skipping media analysis for this URL.")
                        continue

                    media_part = {"mime_type": mime_type, "data": media_content}
                    multimodal_prompt = [media_part, "Analyze this media for any clues, secrets, or important information related to a multi-step task or quiz. Extract any text, codes, or instructions. Respond concisely with only the extracted clue."]
                    logging.info(f"Sending media to multimodal LLM. Prompt (text part truncated to 100 chars): {multimodal_prompt[1][:100]}")

                    multimodal_response = _generate_content_with_retry(multimodal_model, multimodal_prompt)
                    clue = multimodal_response.text.strip()
                    if clue:
                        additional_context += f"\n\nClue from {media_url}: {clue}"
                        logging.info(f"Extracted clue from media: {clue[:100]}")
                    else:
                        logging.info(f"No clue extracted from media: {media_url}")

                except Exception as e:
                    logging.error(f"Error processing media {media_url}: {e}")

    initial_payload_str = json.dumps(initial_payload, indent=2)
    prompt = f"""
    You are an intelligent agent tasked with solving a multi-step challenge presented on a webpage. Your goal is to analyze the provided HTML content and create a JSON plan to accomplish the main objective.

    **Instructions:**

    1.  **Analyze the Entire Context:** Read the text, look at all links (`<a>` tags), and identify any embedded media (`<audio>`, `<video>`). The solution may require combining information from the text with data from a linked file (e.g., a CSV, PDF) or transcribed media.
    2.  **Determine the Primary Action:** Decide on the single most important action needed to solve the task. This usually involves fetching and processing data from a URL.
    3.  **Code Execution (STRICTLY FOR COMPUTATION/DATA MANIPULATION):**
        * **Use the Python Code Block:** ONLY use the `"execute_python_code:"` format for tasks requiring custom Python logic, numerical calculation, data filtering/aggregation (e.g., summing columns in a CSV, complex regex on a large text blob). The execution environment has modules like `pandas`, `numpy`, `csv`, `io`, and `BeautifulSoup` available.
        * **CSV Processing Note:** When using `pandas.read_csv`, be aware that the first row is often data, not a header. To ensure all rows are included in calculations, load the data without assuming a header (e.g., use `header=None` and reference columns by index like `df[0]`).
        * **Do NOT Use the Python Code Block:** Do NOT use the `"execute_python_code:"` format for simple extraction or audio transcription.

    4.  **Formulate a Plan:** Respond ONLY with a JSON object with a single key "plan". This object must contain:
        * `task_type`: This can be "fetch_and_submit" for general data retrieval.
        * `fetch_url`: The single, most relevant URL to fetch for this step of the task.
        * `processing_task`: A concise description of the task.
            
            *A. COMPUTATIONAL TASK (USE CODE):*

            *B. NON-COMPUTATIONAL TASK (USE STRING):* The `processing_task` must contain ONLY a descriptive phrase.
            
            Example: `"transcribe_audio_to_find_the_secret_code"` or `"extract_the_CEO_name_from_the_PDF_content"`

        * `submit_url`: The full URL where the final answer should be POSTed.
        * `payload`: The JSON payload for the submission, using "__ANSWER__" as a placeholder for the result of the `processing_task`.

    **Security Warning:** Be extremely cautious when generating code. Do not generate code that performs destructive operations, accesses sensitive information, or attempts to bypass security measures. Prioritize safe and predictable code generation.

    **Your Context:**
    - You will be provided with the "Initial JSON Payload" which contains necessary data like your email and the current URL.
    - You will be provided with the full "Webpage Content (HTML)".
    - **Available Python Libraries for `execute_python_code`**: `json`, `csv`, `re`, `pandas` (aliased as `pd`), `numpy` (aliased as `np`), `scipy`, `io`, and `BeautifulSoup`.
    {additional_context}

    Initial JSON Payload (Your context): --- {initial_payload_str} ---
    Webpage Content (Your instructions): --- {quiz_content} ---
    """
    logging.info(f"Prompt for generate_plan (truncated to 500 chars): {prompt[:500]}")
    try:
        response = _generate_content_with_retry(model, prompt)
        logging.info(f"Received response from Gemini API (generate_plan): {response.text[:500]}")
    except (genai.types.BlockedPromptException, google.api_core.exceptions.GoogleAPICallError) as e:
        logging.warning(f"Primary model failed ({e}). Attempting fallback to gemini-2.5-flash-lite.")
        model = get_gemini_model(model_name="gemini-2.5-flash-lite")
        response = _generate_content_with_retry(model, prompt)
        logging.info(f"Received response from fallback Gemini API (generate_plan): {response.text[:500]}")

    logging.info("Exiting generate_plan function.")
    return json.dumps({"llm_response": response.text})

def extract_answer(content: str, question: str):
    logging.info("Entering extract_answer function.")
    model = get_gemini_model()
    if model is None:
        logging.error("Gemini model initialization failed: get_gemini_model returned None for extract_answer.")
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
    """
    logging.info(f"Prompt for extract_answer (truncated to 500 chars): {prompt[:500]}")
    try:
        response = _generate_content_with_retry(model, prompt)
        logging.info(f"Received response from Gemini API (extract_answer): {response.text[:500]}")
    except (genai.types.BlockedPromptException, google.api_core.exceptions.GoogleAPICallError) as e:
        logging.warning(f"Primary model failed ({e}). Attempting fallback to gemini-2.5-flash-latest.")
        model = get_gemini_model(model_name="gemini-2.5-flash-latest")
        response = _generate_content_with_retry(model, prompt)
        logging.info(f"Received response from fallback Gemini API (extract_answer): {response.text[:500]}")

    logging.info("Exiting extract_answer function.")
    return json.dumps({"answer": response.text.strip()})

def process_data(data_content: str, processing_task: str):
    logging.info("Entering process_data function.")
    model = get_gemini_model()
    if model is None:
        logging.error("Gemini model initialization failed: get_gemini_model returned None for process_data.")
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
    """
    logging.info(f"Prompt for process_data (truncated to 500 chars): {prompt[:500]}")
    try:
        response = _generate_content_with_retry(model, prompt)
        logging.info(f"Received response from Gemini API (process_data): {response.text[:500]}")
    except (genai.types.BlockedPromptException, google.api_core.exceptions.GoogleAPICallError) as e:
        logging.warning(f"Primary model failed ({e}). Attempting fallback to gemini-2.5-flash-latest.")
        model = get_gemini_model(model_name="gemini-2.5-flash-latest")
        response = _generate_content_with_retry(model, prompt)
        logging.info(f"Received response from fallback Gemini API (process_data): {response.text[:500]}")

    logging.info("Exiting process_data function.")
    return json.dumps({"processed_data": response.text.strip()})

def process_audio_with_gemini(audio_content: bytes, processing_task: str):
    logging.info("Entering process_audio_with_gemini function.")
    model = get_gemini_model()
    if model is None:
        logging.error("Gemini model initialization failed: get_gemini_model returned None for process_audio_with_gemini.")
        raise RuntimeError("Gemini model initialization failed.")
    
    audio_part = {
        "mime_type": "audio/ogg",
        "data": audio_content
    }

    prompt = [audio_part, f"Analyze the provided audio and {processing_task}. Respond with ONLY the final answer, and nothing else."]
    logging.info(f"Prompt for process_audio_with_gemini (text part truncated to 500 chars): {prompt[1][:500]}")
    logging.info(f"Audio content size for process_audio_with_gemini: {len(audio_content)} bytes.")

    try:
        response = _generate_content_with_retry(model, prompt)
        logging.info(f"Received response from Gemini API (process_audio_with_gemini): {response.text[:500]}")
    except (genai.types.BlockedPromptException, google.api_core.exceptions.GoogleAPICallError) as e:
        logging.warning(f"Primary model failed ({e}). Attempting fallback to gemini-2.5-flash-latest.")
        model = get_gemini_model(model_name="gemini-2.5-flash-latest")
        response = _generate_content_with_retry(model, prompt)
        logging.info(f"Received response from fallback Gemini API (process_audio_with_gemini): {response.text[:500]}")

    logging.info("Exiting process_audio_with_gemini function.")
    return json.dumps({"processed_data": response.text.strip()})

if __name__ == "__main__":
    async def main_async():
        try:
            input_data = json.load(sys.stdin)
            task = input_data.get("task")

            if task == "generate_plan":
                result_json = await generate_plan(input_data["quiz_content"], input_data["initial_payload"], input_data.get("media_urls"))
                print(result_json)
            
            elif task == "extract_answer":
                result_json = extract_answer(input_data["content"], input_data["question"])
                print(result_json)
                
            elif task == "process_data":
                result_json = process_data(input_data["data_content"], input_data["processing_task"])
                print(result_json)
            
            elif task == "process_audio_with_gemini":
                audio_content_base64 = input_data["audio_content"]
                audio_bytes = base64.b64decode(audio_content_base64)
                result_json = process_audio_with_gemini(audio_bytes, input_data["processing_task"])
                print(result_json)

            else:
                raise ValueError(f"Unknown task: {task}")
            
        except Exception as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            sys.exit(1)

    asyncio.run(main_async())