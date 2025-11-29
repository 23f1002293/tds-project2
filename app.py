import os
import json
import asyncio
import logging
import sys
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import traceback
from playwright.async_api import async_playwright
import httpx
import mimetypes # For file type detection
from urllib.parse import urlparse, urljoin
from tenacity import retry, stop_after_attempt, wait_fixed # Import tenacity
from bs4 import BeautifulSoup
import base64
import re
# Removed redundant data_processor import
import tempfile
import uuid
import datetime
from contextlib import redirect_stdout
import io
# Imported libraries for the exec sandbox:
import pandas 
import numpy 
import scipy 
import csv 

# Configure logging to flush immediately
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Ensure the StreamHandler flushes immediately
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.flush = sys.stdout.flush

# -----------------------------------------------------------
# MERGED PARSER FUNCTIONS (From parsers.py - Kept for reference, but not explicitly used in processing)
# NOTE: The LLM should still be instructed to use pandas/BeautifulSoup directly for safety.
# -----------------------------------------------------------

# def parse_pdf(content: bytes) -> str:
#     # ... (Logic from parsers.py)
#     pass

# def parse_csv(content: str) -> list[list[str]]:
#     # ... (Logic from parsers.py)
#     pass

# -----------------------------------------------------------


def generate_temp_file_name(original_extension):
    """Generates a unique temporary file name.
    Format: LLM_generated_fileYYYYMMDDHHMMSS_UUIDsuffix.ext
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    unique_suffix = str(uuid.uuid4())[:8] # Short UUID for uniqueness
    return f"LLM_generated_file{timestamp}_{unique_suffix}{original_extension}"

def _execute_generated_code(code_string: str, data_content: str, html_content: str) -> str:
    """
    Safely executes the generated Python code within an isolated scope (sandbox).
    The code must print the final answer to stdout.
    """
    
    # --- START SECURITY ENHANCEMENT: Injects safe built-ins for imports/basic logic ---
    safe_builtins = __builtins__.copy()

    # Define UNSAFE built-ins to remove or replace
    unsafe_builtins = [
        'open', 'exit', 'quit', 'eval', 'exec', 'compile', 'input', 
        'copyright', 'credits', 'license', 'breakpoint'
    ]

    # Remove unsafe built-ins from the execution context
    for unsafe_func in unsafe_builtins:
        if unsafe_func in safe_builtins:
            del safe_builtins[unsafe_func]
            
    # Define local variables/modules available to the generated code
    local_vars = {
        # Inject safe, pre-installed modules
        'json': json,
        'csv': csv, 
        're': re,
        'pandas': pandas,
        'pd': pandas, # Convenient alias for LLM-generated code
        'numpy': numpy,
        'np': numpy,   # Convenient alias
        'scipy': scipy,
        'io': io,
        'BeautifulSoup': BeautifulSoup,  # Expose BeautifulSoup
        'data_content': data_content,    # The fetched data (e.g., CSV string)
        'html_content': html_content     # The initial page HTML string
    }
    # --- END SECURITY ENHANCEMENT ---


    # We wrap the execution to capture stdout.
    stdout_capture = io.StringIO()
    
    try:
        logging.info(f"--- EXECUTING GENERATED PYTHON CODE ---\n{code_string}\n---------------------------------------")
        # Execute the code, capturing stdout. 
        with redirect_stdout(stdout_capture):
            # The global context is the filtered built-ins; the local context has the modules/data
            exec(code_string, {"__builtins__": safe_builtins}, local_vars)

        result = stdout_capture.getvalue().strip()
        
        if not result:
            raise ValueError("Execution succeeded, but no output was printed.")
        
        logging.info(f"--- EXECUTION RESULT ---\n{result}\n--------------------------")
        return result

    except Exception as e:
        # Return a clear error if execution fails
        raise RuntimeError(f"Code execution failed: {e}")


async def homepage(request):
    logging.info(f"Incoming request: {request.url} from {request.client.host}")
    logging.info("Homepage endpoint called")
    results = []
    task_counter = 0
    
    try:
        data = await request.json()
        logging.info(f"Initial request data: {data}")
    except json.JSONDecodeError:
        logging.error("Invalid JSON received")
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    email, secret, current_url = data.get('email'), data.get('secret'), data.get('url')
    if not all([email, secret, current_url]):
        logging.warning("Missing required fields in initial request")
        return JSONResponse({"error": "Missing required fields"}, status_code=400)
    
    app_secret = os.environ.get('APP_SECRET')
    if not app_secret:
        logging.error("APP_SECRET environment variable is not set.")
        return JSONResponse({"error": "Server configuration error"}, status_code=500)
    
    if secret != app_secret:
        logging.warning("Invalid secret provided from client.")
        return JSONResponse({"error": "Invalid secret"}, status_code=403)

    # Increased timeout from 60.0s to 540.0s (9 minutes) for long Playwright/HTTP operations
    timeout = httpx.Timeout(200.0) 
    async with httpx.AsyncClient(timeout=timeout) as client:
        while current_url:
            task_counter += 1
            logging.info(f"--- Starting Task {task_counter} for URL: {current_url} ---")
            
            try:
                # --- A: FETCH INITIAL PAGE FOR CLUES ---
                logging.info("A: Attempting to launch Playwright for instructions")
                async with async_playwright() as p:
                    logging.info("A.1: Playwright launched.")
                    logging.info("A.2: Launching Chromium browser.")
                    browser = await p.chromium.launch()
                    logging.info("A.3: Browser launched. Creating new page.")
                    page = await browser.new_page()
                    logging.info("A.4: New page created.")
                    await page.goto(current_url, wait_until='domcontentloaded')
                    logging.info("C: Navigation complete. Retrieving page content.")
                    page_content = await page.content()
                    logging.info("C.1: Page content retrieved.")

                    
                    # Pre-process HTML to resolve relative URLs
                    soup = BeautifulSoup(page_content, 'html.parser')
                    for tag in soup.find_all(href=True):
                        tag['href'] = urljoin(current_url, tag['href'])
                    for tag in soup.find_all(src=True):
                        tag['src'] = urljoin(current_url, tag['src'])
                    
                    processed_html = str(soup) # <-- This is the HTML from the current page
                    logging.info(f"--- Processed HTML with Absolute URLs ---\n{processed_html[:1000]}\n---------------------------------")

                    # Extract audio/video URLs
                    media_urls = []
                    for tag in soup.find_all(['audio', 'video']):
                        if tag.get('src'):
                            media_url = urljoin(current_url, tag['src'])
                            media_urls.append(media_url)
                    
                    if media_urls:
                        logging.info(f"Found media URLs: {media_urls}")
                    else:
                        logging.info("No media URLs found on the page.")

                    await browser.close()
                logging.info("D: Playwright operations completed successfully.")

                # --- E: PLAN GENERATION ---
                logging.info("E: Preparing input for Gemini worker.")
                worker_input = json.dumps({
                    "task": "generate_plan",
                    "quiz_content": processed_html.strip(),
                    "initial_payload": {"email": email, "secret": secret, "url": current_url},
                    "media_urls": media_urls
                }).encode('utf-8')

                python_executable = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
                if not os.path.exists(python_executable):
                    python_executable = "python"
                
                # Timeout for worker process (Plan Generation) increased from 180s to 540s
                proc = await asyncio.create_subprocess_exec(
                    python_executable, 'gemini_worker.py',
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                logging.info("F: Invoking Gemini worker for plan generation.")
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(input=worker_input), timeout=540)
                except asyncio.TimeoutError:
                    logging.error("Gemini worker (generate_plan) timed out after 9 minutes.")
                    proc.kill()
                    await proc.wait()
                    raise RuntimeError("Gemini worker timed out during plan generation.")
                logging.info("G: Gemini worker for plan generation completed.")

                if proc.returncode != 0:
                    error_message = stderr.decode('utf-8')
                    logging.error(f"Gemini worker script failed with exit code {proc.returncode}: {error_message}")
                    raise RuntimeError(f"Gemini worker script failed: {error_message}")

                worker_output = json.loads(stdout.decode('utf-8'))
                logging.info("H: Successfully received output from Gemini worker.")

                llm_response_str = worker_output.get("llm_response", "{}")
                start_index = llm_response_str.find('{')
                end_index = llm_response_str.rfind('}')
                if start_index == -1 or end_index == -1:
                    raise json.JSONDecodeError("No JSON object found in llm_response", llm_response_str, 0)
                
                llm_response_json = json.loads(llm_response_str[start_index:end_index+1])
                plan = llm_response_json.get("plan")
                if not plan:
                    raise ValueError("No plan found in LLM response")

                logging.info(f"I: Executing plan: {plan}")
                
                task_type = plan.get("task_type")
                submit_url = plan.get("submit_url")
                payload = plan.get("payload")
                processing_task = plan.get("processing_task")
                
                final_payload = payload
                processed_answer = None
                
                # Default content for code execution sandbox is the HTML content 
                content_for_code_execution = processed_html

                if task_type == "fetch_and_submit":
                    fetch_url = plan.get("fetch_url")
                    logging.info(f"J: Initial Fetching data from: {fetch_url}")

                    fetched_content_bytes = None
                    fetched_content_text = None
                    actual_file_extension = os.path.splitext(urlparse(fetch_url).path)[1].lower()
                    
                    # --- FETCH LOGIC (HTTP) ---
                    if actual_file_extension in [
                        '.pdf', '.docx', '.mp3', '.opus', '.wav', '.flac', '.ogg', '.m4a', '.mp4', '.avi', '.mov', '.webm',
                        '.csv', '.json', '.xml'
                    ]:
                        logging.info(f"Attempting direct HTTP fetch for file type: {actual_file_extension}")
                        fetch_response = await client.get(fetch_url)
                        fetch_response.raise_for_status() 
                        logging.info(f"Direct HTTP fetch successful for {actual_file_extension} from {fetch_url}")
                        
                        if actual_file_extension in ['.mp3', '.opus', '.wav', '.flac', '.ogg', '.m4a', '.mp4', '.avi', '.mov', '.webm']:
                            fetched_content_bytes = fetch_response.content
                            fetched_content_text = fetched_content_bytes.decode('utf-8', errors='ignore')
                        else: 
                            fetched_content_text = fetch_response.text

                    else: # Assume HTML or dynamic content, use Playwright
                        logging.info(f"Attempting Playwright fetch for dynamic content from {fetch_url}")
                        async with async_playwright() as p:
                            browser = await p.chromium.launch()
                            page = await browser.new_page()
                            await page.goto(fetch_url, wait_until='domcontentloaded')
                            fetched_content_text = await page.locator('body').text_content()
                            logging.info(f"--- Rendered content of fetch_url ---\n{fetched_content_text[:500]}\n------------------------------------ (Playwright)")
                            await browser.close()
                        logging.info("Fetched dynamic content with Playwright successfully.")
                    
                    # If fetching a file (like CSV), that file content becomes the primary data for execution
                    if fetched_content_text is not None:
                         content_for_code_execution = fetched_content_text
                    
                    
                    # --- PROCESSING LOGIC ---

                    # 1. AUDIO PROCESSING (Priority 1)
                    if actual_file_extension in ['.mp3', '.opus', '.wav', '.flac', '.ogg', '.m4a']:
                        if not fetched_content_bytes:
                            raise ValueError("No audio content found for processing.")

                        audio_content_base64 = base64.b64encode(fetched_content_bytes).decode('utf-8')
                        
                        logging.info("1: Calling multimodal Gemini worker to process audio directly.")
                        processing_input = json.dumps({
                            "task": "process_audio_with_gemini",
                            "audio_content": audio_content_base64,
                            "processing_task": processing_task 
                        }).encode('utf-8')
                        
                    
                    # 2. CODE EXECUTION (Priority 2: Complex Computation)
                    # Task 2 (Fixed format) has been removed.
                    if processing_task and "execute_python_code" in processing_task and processed_answer is None:
                        
                        code_match = re.search(r"```python\s*([\s\S]*?)```", processing_task)
                        if code_match:
                            generated_code = code_match.group(1).strip()
                            logging.info("2: Executing LLM-generated Python code locally (Async Thread).")
                            
                            # Run synchronous exec() in a separate thread to prevent blocking the event loop
                            processed_answer = await asyncio.to_thread(
                                _execute_generated_code, generated_code, content_for_code_execution, processed_html
                            )
                            
                        else:
                             # If code execution was specified but code block is missing, fall through to generic LLM call (Block 3)
                             pass

                    # 3. GENERIC LLM PROCESSING (Priority 3: Text Extraction / Fallback)
                    if processed_answer is None:
                        content_for_processing = fetched_content_text if fetched_content_text is not None else ""
                        logging.info(f"3: Calling Gemini worker for generic data processing: {processing_task}")
                        processing_input = json.dumps({
                            "task": "process_data",
                            "data_content": content_for_processing,
                            "processing_task": processing_task
                        }).encode('utf-8')
                        
                        # Timeout for worker process (Data Processing) increased from 180s to 540s
                        proc_process = await asyncio.create_subprocess_exec(
                            python_executable, 'gemini_worker.py',
                            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                        )
                        try:
                            process_stdout, process_stderr = await asyncio.wait_for(proc_process.communicate(input=processing_input), timeout=540)
                        except asyncio.TimeoutError:
                            logging.error("Gemini text processing worker timed out after 9 minutes.")
                            proc_process.kill()
                            await proc_process.wait()
                            raise RuntimeError("Gemini text processing timed out.")
                        
                        if proc_process.returncode != 0:
                            raise RuntimeError(f"Gemini processing worker failed: {process_stderr.decode('utf-8')}")
                        
                        processing_output = json.loads(process_stdout.decode('utf-8'))
                        processed_answer = processing_output.get("processed_data")
                    
                    
                    logging.info(f"Processed answer: {processed_answer}")
                    
                    def replace_placeholder(p, answer):
                        if isinstance(p, dict):
                            return {k: replace_placeholder(v, answer) for k, v in p.items()}
                        elif isinstance(p, list):
                            return [replace_placeholder(i, answer) for i in p]
                        elif isinstance(p, str) and p == "__ANSWER__":
                            return answer
                        else:
                            return p
                    
                    final_payload = replace_placeholder(payload, processed_answer)

                # Centralized submission logic with retry
                logging.info(f"L: Submitting final payload: {json.dumps(final_payload)} to {submit_url}")

                @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
                async def post_with_retry():
                    try:
                        logging.info(f"Attempting POST to {submit_url}...")
                        return await client.post(submit_url, json=final_payload)
                    except httpx.RemoteProtocolError as e:
                        logging.warning(f"Caught RemoteProtocolError, retrying... Error: {e}")
                        raise
                
                response = await post_with_retry()
                
                logging.info("M: Submission response received")
                command_output_json = response.json()

                logging.info(f"N: Submission successful. Response: {command_output_json}")
                iteration_result = {
                    "task_number": task_counter,
                    "url_processed": current_url,
                    "llm_plan": plan,
                    "final_payload": final_payload,
                    "submission_response": command_output_json
                }
                results.append(iteration_result)
                
                current_url = command_output_json.get("url")

            except Exception as e:
                logging.error(f"An error occurred during Task {task_counter} for {current_url}: {e}")
                traceback.print_exc()
                results.append({"task_number": task_counter, "url_processed": current_url, "error": str(e)})
                current_url = None
    
    logging.info("--- Workflow finished ---")
    final_response_payload = {
        "message": "Workflow complete",
        "tasks_completed": task_counter,
        "results": results
    }
    logging.info(f"Returning final JSON response: {json.dumps(final_response_payload)}")
    return JSONResponse(final_response_payload, status_code=200)

routes = [
    Route("/", homepage, methods=["POST"]),
]

app = Starlette(routes=routes)