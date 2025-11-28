import os
import json
import asyncio
import logging
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import traceback
from playwright.async_api import async_playwright
import httpx
import mimetypes # For file type detection
from urllib.parse import urlparse
import speech_recognition as sr
from pydub import AudioSegment
from tenacity import retry, stop_after_attempt, wait_fixed # Import tenacity

import parsers
import data_processor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def transcribe_audio(audio_content_bytes: bytes, audio_format: str) -> str:
    """Transcribes audio content to text."""
    recognizer = sr.Recognizer()
    temp_audio_file = f"/tmp/audio_to_transcribe.{audio_format}"
    temp_wav_file = "/tmp/audio_to_transcribe.wav"

    try:
        # Save the audio content to a temporary file
        with open(temp_audio_file, "wb") as f:
            f.write(audio_content_bytes)
        
        # Convert to WAV using pydub, as SpeechRecognition prefers WAV
        audio = AudioSegment.from_file(temp_audio_file, format=audio_format)
        audio.export(temp_wav_file, format="wav")

        with sr.AudioFile(temp_wav_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data) # Using Google Web Speech API
            logging.info(f"Transcribed audio: {text}")
            return text
    except Exception as e:
        logging.error(f"Error during audio transcription: {e}")
        return f"Transcription Error: {e}"
    finally:
        # Clean up temporary files
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
        if os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)

async def homepage(request):
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
    if secret != app_secret:
        logging.warning("Invalid secret provided")
        return JSONResponse({"error": "Invalid secret"}, status_code=403)

    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while current_url:
            task_counter += 1
            logging.info(f"--- Starting Task {task_counter} for URL: {current_url} ---")
            
            try:
                logging.info("A: Launching Playwright for instructions")
                async with async_playwright() as p:
                    browser = await p.chromium.launch()
                    page = await browser.new_page()
                    logging.info(f"B: Navigating to URL: {current_url}")
                    await page.goto(current_url, wait_until='domcontentloaded')
                    logging.info("C: Navigation complete")
                    page_content = await page.content()
                    logging.info(f"--- Playwright Scraped Content ---\n{page_content}\n---------------------------------")
                    await browser.close()
                logging.info("D: Playwright operations completed successfully")

                logging.info("E: Calling Gemini worker for a plan")
                worker_input = json.dumps({
                    "task": "generate_plan",
                    "quiz_content": page_content.strip(),
                    "initial_payload": {"email": email, "secret": secret, "url": current_url}
                }).encode('utf-8')

                python_executable = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
                if not os.path.exists(python_executable):
                    python_executable = "python"
                
                proc = await asyncio.create_subprocess_exec(
                    python_executable, 'gemini_worker.py',
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                logging.info("F: Waiting for Gemini worker to complete")
                stdout, stderr = await proc.communicate(input=worker_input)
                logging.info("G: Gemini worker finished")

                if proc.returncode != 0:
                    raise RuntimeError(f"Gemini worker script failed: {stderr.decode('utf-8')}")

                worker_output = json.loads(stdout.decode('utf-8'))
                logging.info("H: Gemini worker script executed successfully")

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
                    logging.info(f"J: Fetching data from: {fetch_url}")

                    parsed_url = urlparse(fetch_url)
                    file_extension = os.path.splitext(parsed_url.path)[1].lower()
                    
                    fetched_content_bytes = None
                    fetched_content_text = None

                    if file_extension in [
                        '.pdf', '.docx', '.mp3', '.opus', '.wav', '.flac', '.ogg', '.m4a', '.mp4', '.avi', '.mov', '.webm',
                        '.csv', '.json', '.xml'
                    ]:
                        fetch_response = await client.get(fetch_url)
                        if file_extension in ['.pdf', '.docx', '.mp3', '.opus', '.wav', '.flac', '.ogg', '.m4a', '.mp4', '.avi', '.mov', '.webm']:
                            fetched_content_bytes = fetch_response.content
                            logging.info(f"Fetched binary content ({len(fetched_content_bytes)} bytes) for {file_extension} file.")
                            # For binary files, text content is derived if needed for LLM, but raw bytes are primary
                            fetched_content_text = fetched_content_bytes.decode('utf-8', errors='ignore')
                        else: # .csv, .json, .xml fetched as text
                            fetched_content_text = fetch_response.text
                            logging.info(f"Fetched text content ({len(fetched_content_text)} chars) for {file_extension} file.")

                    else: # Assume HTML or dynamic content, use Playwright
                        async with async_playwright() as p:
                            browser = await p.chromium.launch()
                            page = await browser.new_page()
                            await page.goto(fetch_url, wait_until='domcontentloaded')
                            fetched_content_text = await page.locator('body').text_content()
                            logging.info(f"--- Rendered content of fetch_url ---\n{fetched_content_text}\n------------------------------------ (Playwright)")
                            await browser.close()
                        logging.info("Fetched dynamic content with Playwright.")
                    
                    # Log all fetched content for debugging, as requested
                    log_content = fetched_content_text if fetched_content_text is not None else (f"Binary content of {len(fetched_content_bytes)} bytes" if fetched_content_bytes else 'None')
                    logging.info(f"--- Fetched content for processing (raw) ---\n{log_content[:500]}\n------------------------------------")
                    
                    processed_answer = None
                    if processing_task == "transcribe_audio":
                        if not fetched_content_bytes:
                            raise ValueError("No audio content found for transcription.")

                        audio_format = file_extension.lstrip('.')
                        if not audio_format:
                            content_type_from_header = fetch_response.headers.get('content-type') if 'fetch_response' in locals() else None
                            if content_type_from_header and 'audio/' in content_type_from_header:
                                audio_format = content_type_from_header.split('/')[-1] 
                            else:
                                logging.warning(f"Could not determine audio format for {fetch_url}, defaulting to 'opus'.")
                                audio_format = 'opus' 

                        transcribed_text = await transcribe_audio(fetched_content_bytes, audio_format)
                        logging.info(f"Transcribed audio: {transcribed_text}")

                        logging.info("Calling Gemini worker to process transcribed audio.")
                        processing_input = json.dumps({
                            "task": "process_data",
                            "data_content": transcribed_text,
                            "processing_task": "extract_clue_from_transcript" 
                        }).encode('utf-8')

                        proc_process = await asyncio.create_subprocess_exec(
                            python_executable, 'gemini_worker.py',
                            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                        )
                        process_stdout, process_stderr = await proc_process.communicate(input=processing_input)
                        if proc_process.returncode != 0:
                            raise RuntimeError(f"Gemini processing worker failed with transcribed audio: {process_stderr.decode('utf-8')}")

                        processing_output = json.loads(process_stdout.decode('utf-8'))
                        processed_answer = processing_output.get("processed_data")

                    elif processing_task:
                        # For other processing tasks, use the fetched text content.
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
                        process_stdout, process_stderr = await proc_process.communicate(input=processing_input)
                        if proc_process.returncode != 0:
                            raise RuntimeError(f"Gemini processing worker failed: {process_stderr.decode('utf-8')}")
                        
                        processing_output = json.loads(process_stdout.decode('utf-8'))
                        processed_answer = processing_output.get("processed_data")

                    else:
                        # Default fallback: if no specific processing task, use appropriate fetched text content
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
    return JSONResponse({
        "message": "Workflow complete",
        "tasks_completed": task_counter,
        "results": results
    }, status_code=200)

routes = [
    Route("/", homepage, methods=["POST"]),
]

app = Starlette(routes=routes)