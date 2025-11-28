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
import data_processor
import tempfile
import uuid
import datetime

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


def generate_temp_file_name(original_extension):
    """Generates a unique temporary file name.
    Format: LLM_generated_fileYYYYMMDDHHMMSS_UUIDsuffix.ext
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    unique_suffix = str(uuid.uuid4())[:8] # Short UUID for uniqueness
    return f"LLM_generated_file{timestamp}_{unique_suffix}{original_extension}"


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

    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while current_url:
            task_counter += 1
            logging.info(f"--- Starting Task {task_counter} for URL: {current_url} ---")
            
            try:
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
                    logging.info(f"--- Playwright Scraped Content (Original) ---\n{page_content[:1000]}\n---------------------------------")
                    
                    # Pre-process HTML to resolve relative URLs
                    soup = BeautifulSoup(page_content, 'html.parser')
                    for tag in soup.find_all(href=True):
                        tag['href'] = urljoin(current_url, tag['href'])
                    for tag in soup.find_all(src=True):
                        tag['src'] = urljoin(current_url, tag['src'])
                    
                    processed_html = str(soup)
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
                
                proc = await asyncio.create_subprocess_exec(
                    python_executable, 'gemini_worker.py',
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                logging.info("F: Invoking Gemini worker for plan generation.")
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(input=worker_input), timeout=180)
                except asyncio.TimeoutError:
                    logging.error("Gemini worker (generate_plan) timed out after 3 minutes.")
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



                if task_type == "fetch_and_submit":
                    fetch_url = plan.get("fetch_url")
                    logging.info(f"J: Initial Fetching data from: {fetch_url}")

                    # --- New: Centralized fetching and content determination ---
                    # This block will handle fetching the content based on the *actual* fetch_url
                    # and determining if it's binary or text.
                    fetched_content_bytes = None
                    fetched_content_text = None
                    actual_file_extension = os.path.splitext(urlparse(fetch_url).path)[1].lower()
                    
                    if actual_file_extension in [
                        '.pdf', '.docx', '.mp3', '.opus', '.wav', '.flac', '.ogg', '.m4a', '.mp4', '.avi', '.mov', '.webm',
                        '.csv', '.json', '.xml'
                    ]:
                        logging.info(f"Attempting direct HTTP fetch for file type: {actual_file_extension}")
                        fetch_response = await client.get(fetch_url)
                        fetch_response.raise_for_status() # Raise an exception for HTTP errors
                        logging.info(f"Direct HTTP fetch successful for {actual_file_extension} from {fetch_url}")
                        if actual_file_extension in ['.pdf', '.docx', '.mp3', '.opus', '.wav', '.flac', '.ogg', '.m4a', '.mp4', '.avi', '.mov', '.webm']:
                            fetched_content_bytes = fetch_response.content
                            logging.info(f"Fetched binary content ({len(fetched_content_bytes)} bytes) for {actual_file_extension} file.")
                            fetched_content_text = fetched_content_bytes.decode('utf-8', errors='ignore') # For potential LLM use
                        else: # .csv, .json, .xml fetched as text
                            fetched_content_text = fetch_response.text
                            logging.info(f"Fetched text content ({len(fetched_content_text)} chars) for {actual_file_extension} file.")

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
                    
                    # Log all fetched content for debugging
                    log_content = fetched_content_text if fetched_content_text is not None else (f"Binary content of {len(fetched_content_bytes)} bytes" if fetched_content_bytes else 'None')
                    logging.info(f"--- Fetched content for processing (raw) ---\n{log_content[:500]}\n------------------------------------")
                    
                    processed_answer = None

                    # --- Reworked Logic: Prioritize local processing based on task description ---
                    sum_task_match = re.match(r"sum_csv_values_greater_than_(\d+)", processing_task)

                    if sum_task_match:
                        # This is a local computational task, ensure we have the CSV content
                        threshold = int(sum_task_match.group(1))

                        # If the initial fetch_url was not a CSV but the task is to sum CSV, 
                        # we need to find and fetch the CSV from the page content.
                        if actual_file_extension != '.csv' and fetched_content_text:
                            logging.info("LLM requested CSV sum, but initial fetch was not a CSV. Searching for CSV link.")
                            soup_for_csv = BeautifulSoup(processed_html, 'html.parser')
                            csv_link = None
                            for a_tag in soup_for_csv.find_all('a', href=True):
                                if a_tag['href'].lower().endswith('.csv'):
                                    csv_link = a_tag['href']
                                    break
                            
                            if csv_link:
                                logging.info(f"Found CSV link: {csv_link}. Fetching CSV content.")
                                csv_response = await client.get(csv_link)
                                fetched_content_text = csv_response.text # This is the content for local processing
                                actual_file_extension = '.csv' # Update for consistency
                            else:
                                logging.warning("Could not find CSV link on the page for summation task.")
                                # Fallback to sending HTML to LLM, might timeout if LLM tries to sum HTML

                        if actual_file_extension == '.csv' and fetched_content_text:
                            logging.info(f"Handling CSV summation locally for threshold: {threshold}")
                            temp_file_path = os.path.join(tempfile.gettempdir(), f"temp_csv_{uuid.uuid4().hex}.csv")
                            logging.info(f"Writing CSV content to temporary file: {temp_file_path}")
                            with open(temp_file_path, 'w', encoding='utf-8') as temp_csv:
                                temp_csv.write(fetched_content_text)
                            processed_answer = data_processor.sum_csv_values_greater_than(
                                temp_file_path, threshold
                            )
                            logging.info(f"Local CSV processing result: {processed_answer}")
                            os.remove(temp_file_path)
                            logging.info(f"Removed temporary file: {temp_file_path}")
                        else:
                            logging.error("Failed to get CSV content for local summation. Falling back to LLM for processing.")
                            # If we couldn't get CSV content, send the original fetched text to LLM
                            content_for_processing = fetched_content_text if fetched_content_text is not None else ""
                            logging.info(f"Calling Gemini worker for data processing (fallback): {processing_task}")
                            processing_input = json.dumps({
                                "task": "process_data",
                                "data_content": content_for_processing,
                                "processing_task": processing_task
                            }).encode('utf-8')
                            
                            proc_process = await asyncio.create_subprocess_exec(
                                python_executable, 'gemini_worker.py',
                                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                            )
                            try:
                                process_stdout, process_stderr = await asyncio.wait_for(proc_process.communicate(input=processing_input), timeout=180)
                            except asyncio.TimeoutError:
                                logging.error("Gemini text processing worker timed out after 3 minutes (fallback).")
                                proc_process.kill()
                                await proc_process.wait()
                                raise RuntimeError("Gemini text processing timed out (fallback).")
                            
                            if proc_process.returncode != 0:
                                raise RuntimeError(f"Gemini processing worker failed (fallback): {process_stderr.decode('utf-8')}")
                            
                            processing_output = json.loads(process_stdout.decode('utf-8'))
                            processed_answer = processing_output.get("processed_data")

                    elif actual_file_extension in ['.mp3', '.opus', '.wav', '.flac', '.ogg', '.m4a']:
                        # This is an audio file, send to multimodal Gemini worker
                        if not fetched_content_bytes:
                            raise ValueError("No audio content found for processing.")

                        audio_content_base64 = base64.b64encode(fetched_content_bytes).decode('utf-8')
                        
                        logging.info("Calling multimodal Gemini worker to process audio directly.")
                        processing_input = json.dumps({
                            "task": "process_audio_with_gemini",
                            "audio_content": audio_content_base64,
                            "processing_task": processing_task 
                        }).encode('utf-8')
                        
                        proc_process = await asyncio.create_subprocess_exec(
                            python_executable, 'gemini_worker.py',
                            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                        )
                        try:
                            process_stdout, process_stderr = await asyncio.wait_for(proc_process.communicate(input=processing_input), timeout=180)
                        except asyncio.TimeoutError:
                            logging.error("Gemini audio processing worker timed out after 3 minutes.")
                            proc_process.kill()
                            await proc_process.wait()
                            raise RuntimeError("Gemini audio processing timed out.")
                        
                        if proc_process.returncode != 0:
                            raise RuntimeError(f"Gemini audio processing worker failed: {process_stderr.decode('utf-8')}")
                        
                        processing_output = json.loads(process_stdout.decode('utf-8'))
                        processed_answer = processing_output.get("processed_data")

                    elif processing_task:
                        # For other LLM-driven tasks (PDF text analysis, general text extraction etc.)
                        content_for_processing = fetched_content_text if fetched_content_text is not None else ""
                        logging.info(f"Calling Gemini worker for data processing: {processing_task}")
                        processing_input = json.dumps({
                            "task": "process_data",
                            "data_content": content_for_processing,
                            "processing_task": processing_task
                        }).encode('utf-8')
                        
                        proc_process = await asyncio.create_subprocess_exec(
                            python_executable, 'gemini_worker.py',
                            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                        )
                        try:
                            process_stdout, process_stderr = await asyncio.wait_for(proc_process.communicate(input=processing_input), timeout=180)
                        except asyncio.TimeoutError:
                            logging.error("Gemini text processing worker timed out after 3 minutes.")
                            proc_process.kill()
                            await proc_process.wait()
                            raise RuntimeError("Gemini text processing timed out.")
                        
                        if proc_process.returncode != 0:
                            raise RuntimeError(f"Gemini processing worker failed: {process_stderr.decode('utf-8')}")
                        
                        processing_output = json.loads(process_stdout.decode('utf-8'))
                        processed_answer = processing_output.get("processed_data")
                    
                    else:
                        # Default fallback: if no specific processing task, use the fetched text content directly.
                        processed_answer = fetched_content_text if fetched_content_text is not None else ""
                    
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