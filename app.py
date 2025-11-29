import os
import json
import asyncio
import logging
import sys
import time
import traceback
import base64
import re
import uuid
import datetime
import io

from contextlib import redirect_stdout
from urllib.parse import urlparse, urljoin

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from playwright.async_api import async_playwright
import httpx
from tenacity import retry, stop_after_attempt, wait_fixed
from bs4 import BeautifulSoup

# Imported libraries for the exec sandbox:
import pandas
import numpy
import scipy
import csv

# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.flush = sys.stdout.flush

# -------------------------------------------------------------------
# General helpers
# -------------------------------------------------------------------
async def process_with_plan(
    python_executable: str,
    processing_task: str,
    content_for_code_execution: str,
    fetched_content_text: str,
    fetched_content_bytes: bytes,
    actual_file_extension: str,
    processed_html: str,
):
    """
    Run audio processing, code execution, or generic LLM processing as needed.
    Returns (processed_answer, used_python).
    """
    processed_answer = None
    processing_input = None
    used_python = False

    code_requested = bool(processing_task and "execute_python_code" in processing_task)

    # 1. AUDIO PROCESSING
    if actual_file_extension in [
        ".mp3",
        ".opus",
        ".wav",
        ".flac",
        ".ogg",
        ".m4a",
    ]:
        if not fetched_content_bytes:
            raise ValueError("No audio content found for processing.")
        audio_content_base64 = base64.b64encode(fetched_content_bytes).decode("utf-8")
        logging.info("1: Calling multimodal Gemini worker to process audio directly.")
        processing_input = json.dumps(
            {
                "task": "process_audio_with_gemini",
                "audio_content": audio_content_base64,
                "processing_task": processing_task,
            }
        ).encode("utf-8")

    # 2. CODE EXECUTION (required if code_requested)
    if code_requested and processed_answer is None:
        # Extract everything between the first and last ``` fence
        start = processing_task.find("```")
        end = processing_task.rfind("```")

        if start == -1 or end == -1 or end <= start:
            logging.error(
                "Plan requested execute_python_co" \
                "de but no ```"
                "was found. processing_task: %s",
                processing_task,
            )
            raise ValueError(
                "Plan requested execute_python_code but no ``` block "
                "was found in processing_task."
            )

        generated_code = processing_task[start + 3 : end].strip()
        logging.info(
            "2: Executing LLM-generated Python code locally (Async Thread)."
        )
        code_exec_start_time = time.time()
        processed_answer = await asyncio.to_thread(
            _execute_generated_code,
            generated_code,
            content_for_code_execution,
            processed_html,
        )
        used_python = True
        logging.info(
            "2: Code execution completed in "
            f"{time.time() - code_exec_start_time:.2f} seconds."
        )

    # 3. GENERIC LLM PROCESSING (only if no code was requested)
    if processed_answer is None and not code_requested:
        content_for_processing = fetched_content_text if fetched_content_text else ""
        logging.info(
            "3: Calling Gemini worker for generic data processing: "
            f"{processing_task}"
        )
        processing_input = json.dumps(
            {
                "task": "process_data",
                "data_content": content_for_processing,
                "processing_task": processing_task,
            }
        ).encode("utf-8")

    # If we already have an answer (from code) or nothing to send to worker, return
    if processed_answer is not None or processing_input is None:
        return processed_answer, used_python

    # Otherwise call gemini_worker for process_data / process_audio_with_gemini
    process_worker_start_time = time.time()
    proc_process = await asyncio.create_subprocess_exec(
        python_executable,
        "gemini_worker.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        process_stdout, process_stderr = await asyncio.wait_for(
            proc_process.communicate(input=processing_input), timeout=540
        )
    except asyncio.TimeoutError:
        logging.error("Gemini text processing worker timed out after 9 minutes.")
        proc_process.kill()
        await proc_process.wait()
        raise RuntimeError("Gemini text processing timed out.")

    logging.info(
        f"Gemini worker (process_data) STDERR: {process_stderr.decode('utf-8')[:1000]} ..."
    )

    if not process_stdout.strip():
        logging.error("Gemini worker (process_data) returned empty stdout.")
        processing_output = {
            "error": "Gemini worker (process_data) returned no output.",
            "raw_stderr": process_stderr.decode("utf-8"),
        }
    else:
        try:
            processing_output = json.loads(process_stdout.decode("utf-8"))
        except json.JSONDecodeError as e:
            logging.error(
                "Failed to decode JSON from Gemini worker (process_data). "
                f"Error: {e}. Raw stdout: {process_stdout.decode('utf-8')[:1000]}"
            )
            processing_output = {
                "error": "Invalid JSON from Gemini worker (process_data)",
                "raw_stdout": process_stdout.decode("utf-8"),
                "raw_stderr": process_stderr.decode("utf-8"),
            }

    logging.info(
        "3: Gemini worker for data processing completed in "
        f"{time.time() - process_worker_start_time:.2f} seconds."
    )

    if "error" in processing_output:
        raise RuntimeError(
            f"Gemini processing worker reported an error: {processing_output['error']}"
        )

    processed_answer = processing_output.get("processed_data")
    logging.info(f"Processed answer: {processed_answer}")
    return processed_answer, used_python


def generate_temp_file_name(original_extension: str) -> str:
    """
    Generates a unique temporary file name.

    Format: LLM_generated_fileYYYYMMDDHHMMSS_UUIDsuffix.ext
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_suffix = str(uuid.uuid4())[:8]  # Short UUID for uniqueness
    return f"LLM_generated_file{timestamp}_{unique_suffix}{original_extension}"


def _execute_generated_code(
    code_string: str,
    data_content: str,
    html_content: str,
) -> str:
    """
    Executes LLM-generated Python code in a restricted environment.

    The code must print the final answer to stdout.
    The printed output is returned as a string.
    NOTE: This is not a fully secure sandbox; use a separate process/container
    for strong isolation in production.
    """
    if isinstance(__builtins__, dict):
        safe_builtins = dict(__builtins__)
    else:
        safe_builtins = __builtins__.__dict__.copy()

    unsafe_builtins = [
        "open",
        "exit",
        "quit",
        "eval",
        "exec",
        "compile",
        "input",
        "copyright",
        "credits",
        "license",
        "breakpoint",
        # "__import__",
    ]
    for unsafe_name in unsafe_builtins:
        safe_builtins.pop(unsafe_name, None)

    local_vars = {
        "json": json,
        "csv": csv,
        "re": re,
        "pandas": pandas,
        "pd": pandas,
        "numpy": numpy,
        "np": numpy,
        "scipy": scipy,
        "io": io,
        "BeautifulSoup": BeautifulSoup,
        "data_content": data_content,
        "html_content": html_content,
    }

    stdout_capture = io.StringIO()

    try:
        logging.info(
            "--- EXECUTING GENERATED PYTHON CODE ---\n"
            f"{code_string}\n"
            "---------------------------------------"
        )
        with redirect_stdout(stdout_capture):
            exec(code_string, {"__builtins__": safe_builtins}, local_vars)

        result = stdout_capture.getvalue().strip()
        if not result:
            raise ValueError("Execution succeeded, but no output was printed.")

        logging.info(
            "--- EXECUTION RESULT ---\n"
            f"{result}\n"
            "--------------------------"
        )
        return result
    except Exception as e:
        raise RuntimeError(f"Code execution failed: {e}") from e


def replace_placeholder(p, answer):
    """Recursively replace '__ANSWER__' placeholders with the given answer."""
    if isinstance(p, dict):
        return {k: replace_placeholder(v, answer) for k, v in p.items()}
    if isinstance(p, list):
        return [replace_placeholder(i, answer) for i in p]
    if isinstance(p, str) and p == "__ANSWER__":
        return answer
    return p


def force_answer_placeholder(payload):
    """
    Ensure that any 'answer' fields are set to '__ANSWER__'
    so the Python result always overwrites any LLM-guessed value.
    """
    if isinstance(payload, dict):
        new = {}
        for k, v in payload.items():
            if k.lower() == "answer":
                new[k] = "__ANSWER__"
            else:
                new[k] = force_answer_placeholder(v)
        return new
    elif isinstance(payload, list):
        return [force_answer_placeholder(x) for x in payload]
    else:
        return payload


MAX_TASKS = 10  # guard against infinite loops


# -------------------------------------------------------------------
# Async helpers for the LLM workflow
# -------------------------------------------------------------------


async def fetch_page_and_media(browser, current_url: str):
    """Use Playwright to load the page and return processed HTML and media URLs."""
    logging.info("A: Attempting to launch Playwright for instructions")
    playwright_start_time = time.time()

    page = await browser.new_page()
    logging.info("A.4: New page created.")
    await page.goto(current_url, wait_until="domcontentloaded")
    logging.info("C: Navigation complete. Retrieving page content.")
    page_content = await page.content()
    logging.info("C.1: Page content retrieved.")

    soup = BeautifulSoup(page_content, "html.parser")
    for tag in soup.find_all(href=True):
        tag["href"] = urljoin(current_url, tag["href"])
    for tag in soup.find_all(src=True):
        tag["src"] = urljoin(current_url, tag["src"])

    processed_html = str(soup)
    logging.info(
        "--- Processed HTML with Absolute URLs ---\n"
        f"{processed_html[:1000]}\n"
        "---------------------------------"
    )

    media_urls = []
    for tag in soup.find_all(["audio", "video"]):
        if tag.get("src"):
            media_url = urljoin(current_url, tag["src"])
            media_urls.append(media_url)

    if media_urls:
        logging.info(f"Found media URLs: {media_urls}")
    else:
        logging.info("No media URLs found on the page.")

    await page.close()
    logging.info(
        "D: Playwright operations completed successfully in "
        f"{time.time() - playwright_start_time:.2f} seconds."
    )

    return processed_html, media_urls


async def generate_plan_with_worker(
    processed_html: str,
    email: str,
    secret: str,
    current_url: str,
    media_urls: list,
):
    """Call gemini_worker.py to generate an LLM plan."""
    logging.info("E: Preparing input for Gemini worker.")

    worker_input = json.dumps(
        {
            "task": "generate_plan",
            "quiz_content": processed_html.strip(),
            "initial_payload": {"email": email, "secret": secret, "url": current_url},
            "media_urls": media_urls,
        }
    ).encode("utf-8")

    python_executable = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python"
    )
    if not os.path.exists(python_executable):
        python_executable = "python"

    plan_worker_start_time = time.time()
    proc = await asyncio.create_subprocess_exec(
        python_executable,
        "gemini_worker.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    logging.info("F: Invoking Gemini worker for plan generation.")

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=worker_input), timeout=540
        )
    except asyncio.TimeoutError:
        logging.error("Gemini worker (generate_plan) timed out after 9 minutes.")
        proc.kill()
        await proc.wait()
        raise RuntimeError("Gemini worker timed out during plan generation.")

    logging.info(
        "G: Gemini worker for plan generation completed in "
        f"{time.time() - plan_worker_start_time:.2f} seconds."
    )
    logging.info(f"Gemini worker (plan) STDOUT: {stdout.decode('utf-8')[:1000]} ...")
    logging.info(f"Gemini worker (plan) STDERR: {stderr.decode('utf-8')[:1000]} ...")

    worker_stdout_str = stdout.decode("utf-8").strip()
    if not worker_stdout_str:
        raw_stderr = stderr.decode("utf-8")
        logging.error(
            "Gemini worker (plan) returned empty stdout. STDERR:\n%s",
            raw_stderr[:1000],
        )
        worker_output = {
            "error": "Gemini worker (plan) returned no output.",
            "raw_stderr": raw_stderr,
        }
    else:
        try:
            worker_output = json.loads(worker_stdout_str)
        except json.JSONDecodeError as e:
            logging.error(
                "Failed to decode JSON from Gemini worker (plan). "
                f"Error: {e}. Raw stdout: {worker_stdout_str[:1000]}"
            )
            worker_output = {
                "error": "Invalid JSON from Gemini worker (plan)",
                "raw_stdout": worker_stdout_str,
                "raw_stderr": stderr.decode("utf-8"),
            }

    if proc.returncode != 0 and "error" not in worker_output:
        error_message = (
            f"Gemini worker script failed with exit code {proc.returncode}: "
            f"{stderr.decode('utf-8')}"
        )
        logging.error(error_message)
        raise RuntimeError(error_message)

    if "error" in worker_output:
        error_message = worker_output["error"]
        raw_stderr = worker_output.get("raw_stderr", "")
        logging.error(
            "Gemini worker script reported an error: %s. STDERR (truncated): %s",
            error_message,
            raw_stderr[:500],
        )
        raise RuntimeError(
            f"Gemini worker script reported an error: {error_message}."
            f" STDERR: {raw_stderr[:200]}"
        )

    logging.info("H: Successfully received output from Gemini worker.")

    llm_response_str = worker_output.get("llm_response", "{}")
    start_index = llm_response_str.find("{")
    end_index = llm_response_str.rfind("}")
    if start_index == -1 or end_index == -1:
        raise json.JSONDecodeError(
            "No JSON object found in llm_response", llm_response_str, 0
        )

    llm_response_json = json.loads(llm_response_str[start_index : end_index + 1])
    plan = llm_response_json.get("plan")
    if not plan:
        raise ValueError("No plan found in LLM response")

    logging.info(f"I: Executing plan: {plan}")
    return plan, python_executable


async def fetch_plan_data(browser, client, plan, processed_html: str):
    """Fetch any external data required by the plan and return content + metadata."""
    task_type = plan.get("task_type")
    if task_type != "fetch_and_submit":
        raise ValueError(f"Unsupported task_type: {task_type}")

    fetch_url = plan.get("fetch_url")
    logging.info(f"J: Initial Fetching data from: {fetch_url}")

    # NEW: no fetch needed if fetch_url is None/empty
    if not fetch_url:
        logging.info(
            "No fetch_url provided in plan; using current page HTML for code/LLM."
        )
        # content_for_code_execution = current HTML
        # fetched_content_text = None
        # fetched_content_bytes = None
        # actual_file_extension = empty string
        return processed_html, None, None, ""

    fetched_content_bytes = None
    fetched_content_text = None
    content_for_code_execution = processed_html

    actual_file_extension = os.path.splitext(urlparse(fetch_url).path)[1].lower()
    fetch_start_time = time.time()

    if actual_file_extension in [
        ".pdf",
        ".docx",
        ".mp3",
        ".opus",
        ".wav",
        ".flac",
        ".ogg",
        ".m4a",
        ".mp4",
        ".avi",
        ".mov",
        ".webm",
        ".csv",
        ".json",
        ".xml",
    ]:
        logging.info(
            f"Attempting direct HTTP fetch for file type: {actual_file_extension}"
        )
        fetch_response = await client.get(fetch_url)
        fetch_response.raise_for_status()
        logging.info(
            f"Direct HTTP fetch successful for {actual_file_extension} from {fetch_url}"
        )

        if actual_file_extension in [
            ".mp3",
            ".opus",
            ".wav",
            ".flac",
            ".ogg",
            ".m4a",
            ".mp4",
            ".avi",
            ".mov",
            ".webm",
        ]:
            fetched_content_bytes = fetch_response.content
            fetched_content_text = fetched_content_bytes.decode(
                "utf-8", errors="ignore"
            )
        else:
            fetched_content_text = fetch_response.text
    else:
        logging.info(
            f"Attempting Playwright fetch for dynamic content from {fetch_url}"
        )
        page = await browser.new_page()
        await page.goto(fetch_url, wait_until="domcontentloaded")
        fetched_content_text = await page.locator("body").text_content()
        logging.info(
            "--- Rendered content of fetch_url ---\n"
            f"{fetched_content_text[:500]}\n"
            "------------------------------------ (Playwright)"
        )
        await page.close()
        logging.info("Fetched dynamic content with Playwright successfully.")

    logging.info(
        f"J: Data fetching completed in {time.time() - fetch_start_time:.2f} seconds."
    )

    if fetched_content_text is not None:
        content_for_code_execution = fetched_content_text

    return (
        content_for_code_execution,
        fetched_content_text,
        fetched_content_bytes,
        actual_file_extension,
    )


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
async def post_with_retry(client, submit_url: str, final_payload: dict):
    """Submit final payload with retry on transient protocol errors."""
    try:
        logging.info(f"Attempting POST to {submit_url}...")
        return await client.post(submit_url, json=final_payload)
    except httpx.RemoteProtocolError as e:
        logging.warning(f"Caught RemoteProtocolError, retrying... Error: {e}")
        raise


async def homepage(request):
    start_time = time.time()
    logging.info(f"Incoming request: {request.url} from {request.client.host}")
    logging.info(f"Request Headers: {request.headers}")

    results = []
    task_counter = 0

    try:
        data = await request.json()
        logging.info(f"Initial request  {data}")
    except json.JSONDecodeError:
        logging.error("Invalid JSON received")
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    except Exception as e:
        logging.error(f"Error processing request JSON: {e}")
        return JSONResponse(
            {"error": "Error processing request JSON"}, status_code=500
        )

    email = data.get("email")
    secret = data.get("secret")
    current_url = data.get("url")

    if not all([email, secret, current_url]):
        logging.warning("Missing required fields in initial request")
        return JSONResponse({"error": "Missing required fields"}, status_code=400)

    app_secret = os.environ.get("APP_SECRET")
    if not app_secret:
        logging.error("APP_SECRET environment variable is not set.")
        return JSONResponse(
            {"error": "Server configuration error"}, status_code=500
        )

    if secret != app_secret:
        logging.warning("Invalid secret provided from client.")
        return JSONResponse({"error": "Invalid secret"}, status_code=403)

    timeout = httpx.Timeout(200.0)
    previous_url = None

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                while current_url and task_counter < MAX_TASKS:
                    if current_url == previous_url:
                        logging.warning(
                            f"Next URL {current_url} is same as previous; "
                            "stopping to avoid infinite loop."
                        )
                        break
                    previous_url = current_url

                    task_start_time = time.time()
                    task_counter += 1
                    logging.info(
                        f"--- Starting Task {task_counter} for URL: {current_url} ---"
                    )

                    try:
                        processed_html, media_urls = await fetch_page_and_media(
                            browser, current_url
                        )

                        plan, python_executable = await generate_plan_with_worker(
                            processed_html,
                            email,
                            secret,
                            current_url,
                            media_urls,
                        )

                        task_type = plan.get("task_type")
                        submit_url = plan.get("submit_url")
                        payload = plan.get("payload")
                        processing_task = plan.get("processing_task")

                        if task_type != "fetch_and_submit":
                            logging.warning(
                                f"Unsupported task_type {task_type}; stopping workflow."
                            )
                            break

                        code_requested = bool(
                            processing_task
                            and "execute_python_code" in processing_task
                        )
                        if code_requested:
                            payload = force_answer_placeholder(payload)

                        (
                            content_for_code_execution,
                            fetched_content_text,
                            fetched_content_bytes,
                            actual_file_extension,
                        ) = await fetch_plan_data(
                            browser, client, plan, processed_html
                        )

                        processed_answer, used_python = await process_with_plan(
                            python_executable,
                            processing_task,
                            content_for_code_execution,
                            fetched_content_text,
                            fetched_content_bytes,
                            actual_file_extension,
                            processed_html,
                        )

                        logging.info(
                            f"Processed answer (used_python={used_python}): "
                            f"{processed_answer}"
                        )

                        final_payload = replace_placeholder(
                            payload, processed_answer
                        )
                        logging.info(
                            f"L: Submitting final payload: "
                            f"{json.dumps(final_payload)} to {submit_url}"
                        )

                        submission_start_time = time.time()
                        response = await post_with_retry(
                            client, submit_url, final_payload
                        )
                        logging.info(
                            f"L: Submission completed in "
                            f"{time.time() - submission_start_time:.2f} seconds."
                        )
                        logging.info("M: Submission response received")

                        command_output_json = response.json()
                        logging.info(
                            f"N: Submission successful. Response: {command_output_json}"
                        )

                        iteration_result = {
                            "task_number": task_counter,
                            "url_processed": current_url,
                            "llm_plan": plan,
                            "final_payload": final_payload,
                            "submission_response": command_output_json,
                            "used_python": used_python,
                            "processed_answer": processed_answer,
                        }
                        results.append(iteration_result)

                        next_url = command_output_json.get("url")
                        if not next_url:
                            logging.info(
                                "No next URL returned by server; stopping workflow."
                            )
                            current_url = None
                        else:
                            current_url = next_url

                        logging.info(
                            f"--- Task {task_counter} completed in "
                            f"{time.time() - task_start_time:.2f} seconds ---"
                        )

                    except Exception as e:
                        logging.error(
                            f"An error occurred during Task {task_counter} "
                            f"for {current_url}: {e}"
                        )
                        traceback.print_exc()
                        results.append(
                            {
                                "task_number": task_counter,
                                "url_processed": current_url,
                                "error": str(e),
                            }
                        )
                        current_url = None

            finally:
                await browser.close()

    logging.info(
        f"--- Workflow finished in {time.time() - start_time:.2f} seconds ---"
    )
    final_response_payload = {
        "message": "Workflow complete",
        "tasks_completed": task_counter,
        "results": results,
    }
    logging.info(
        f"Returning final JSON response: {json.dumps(final_response_payload)}"
    )
    return JSONResponse(final_response_payload, status_code=200)


routes = [
    Route("/", homepage, methods=["POST"]),
]

app = Starlette(routes=routes)
