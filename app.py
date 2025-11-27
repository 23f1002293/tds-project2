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

import parsers
import data_processor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def homepage(request):
    logging.info("Homepage endpoint called")
    results = []
    
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

    async with httpx.AsyncClient() as client:
        while current_url:
            logging.info(f"--- Starting new iteration for URL: {current_url} ---")
            
            try:
                logging.info("A: Launching Playwright")
                async with async_playwright() as p:
                    browser = await p.chromium.launch()
                    page = await browser.new_page()
                    logging.info(f"B: Navigating to URL: {current_url}")
                    await page.goto(current_url, wait_until='domcontentloaded')
                    logging.info("C: Navigation complete")
                    body_text = await page.locator('body').text_content()
                    logging.info(f"--- Playwright Scraped Content ---\n{body_text}\n---------------------------------")
                    await browser.close()
                logging.info("D: Playwright operations completed successfully")

                logging.info("E: Calling Gemini worker script")
                worker_input = json.dumps({
                    "quiz_content": body_text.strip(),
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
                processing_task = plan.get("processing_task") # New: Task for data_processor
                
                command_output_json = {}
                final_payload = payload
                processed_answer = None

                if task_type == "fetch_and_submit":
                    fetch_url = plan.get("fetch_url")
                    logging.info(f"J: Fetching data from: {fetch_url}")

                    parsed_url = urlparse(fetch_url)
                    file_extension = os.path.splitext(parsed_url.path)[1].lower()
                    
                    fetched_content_bytes = None
                    fetched_content_text = None

                    # Use httpx for binary files, Playwright for web pages
                    if file_extension in ['.pdf', '.docx']:
                        fetch_response = await client.get(fetch_url)
                        fetched_content_bytes = fetch_response.content
                        logging.info(f"Fetched binary content ({len(fetched_content_bytes)} bytes)")
                    else: # Assume HTML or dynamic content
                        async with async_playwright() as p:
                            browser = await p.chromium.launch()
                            page = await browser.new_page()
                            await page.goto(fetch_url, wait_until='domcontentloaded')
                            fetched_content_text = await page.locator('body').text_content()
                            await browser.close()
                        logging.info("Fetched dynamic content with Playwright")
                    
                    logging.info("K: Fetched data response received")

                    # Parse content based on file type
                    if file_extension == '.pdf' and fetched_content_bytes:
                        parsed_content = parsers.parse_pdf(fetched_content_bytes)
                    elif file_extension == '.docx' and fetched_content_bytes:
                        parsed_content = parsers.parse_docx(fetched_content_bytes)
                    elif file_extension == '.json':
                        parsed_content = parsers.parse_json(fetched_content_text)
                    elif file_extension == '.xml':
                        parsed_content = parsers.parse_xml(fetched_content_text)
                    elif file_extension == '.csv':
                        parsed_content = parsers.parse_csv(fetched_content_text)
                    else:
                        parsed_content = fetched_content_text # Default to text

                    logging.info(f"Fetched and parsed content (type: {type(parsed_content)}): {str(parsed_content)[:200]}...")

                    # Apply processing task if specified
                    if processing_task == "sum_numbers_in_csv":
                        processed_answer = data_processor.sum_numbers_in_csv(fetched_content_text)
                    elif processing_task == "extract_secret_code" and isinstance(parsed_content, str):
                        processed_answer = data_processor.extract_secret_code(parsed_content)
                    else:
                        processed_answer = parsed_content

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
                    logging.info(f"L: Submitting final payload: {final_payload} to {submit_url}")
                    response = await client.post(submit_url, json=final_payload)
                    logging.info("M: Submission response received")
                    command_output_json = response.json()

                elif task_type == "single_post":
                    logging.info(f"L: Submitting payload: {payload} to {submit_url}")
                    response = await client.post(submit_url, json=payload)
                    logging.info("M: Submission response received")
                    command_output_json = response.json()
                
                else:
                    raise ValueError(f"Unknown task type: {task_type}")

                logging.info(f"N: Submission successful. Response: {command_output_json}")
                iteration_result = {
                    "url_processed": current_url,
                    "llm_plan": plan,
                    "final_payload": final_payload,
                    "submission_response": command_output_json
                }
                results.append(iteration_result)
                
                current_url = command_output_json.get("url")

            except Exception as e:
                logging.error(f"An error occurred during iteration for {current_url}: {e}")
                traceback.print_exc()
                results.append({"url_processed": current_url, "error": str(e)})
                current_url = None # Stop the loop
    
    logging.info("--- Workflow finished ---")
    return JSONResponse({
        "message": "Workflow complete",
        "results": results
    }, status_code=200)


routes = [
    Route("/", homepage, methods=["POST"]),
]

app = Starlette(routes=routes)