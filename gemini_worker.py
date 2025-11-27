import os
import json
import sys
import re
import google.generativeai as genai

# This script reads data from stdin, processes it with Gemini, and prints the result to stdout.

def run_gemini_logic(quiz_content: str, initial_payload: dict):
    """
    This function handles all Gemini API logic.
    """
    # 1. Configure API
    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=gemini_api_key)

    # 2. Find the latest model
    latest_version = -1.0
    model_to_use = None
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            match = re.search(r'gemini-(\d+\.?\d*)-pro', model.name)
            if match:
                version = float(match.group(1))
                if version > latest_version:
                    latest_version = version
                    model_to_use = model.name
    if not model_to_use:
        raise RuntimeError("Could not find a suitable Gemini Pro model.")

    # 3. Generate the content
    model = genai.GenerativeModel(model_to_use)
    initial_payload_str = json.dumps(initial_payload, indent=2)
    prompt = f"""
    You are an intelligent agent. Your task is to analyze the webpage content below and create a JSON plan to solve the described task.
    You must use the provided "Initial JSON Payload" to fill in any required data.

    Analyze the task and respond ONLY with a JSON object with a single key "plan".
    The "plan" object should contain:
    1.  `task_type`: A string describing the task. For a task that involves fetching data and then submitting it, use "fetch_and_submit". For a simple POST, use "single_post".
    2.  `fetch_url`: (Only for "fetch_and_submit") The full URL from which to fetch the required data. This should directly point to a data file (e.g., .csv, .txt, .json) if the task requires data processing, otherwise it can be a webpage.
    3.  `processing_task`: (Optional, for "fetch_and_submit") A string indicating how to process the fetched data before replacing `__ANSWER__`. E.g., "sum_numbers_in_csv", "extract_secret_code", "summarize_text", "answer_question".
    4.  `submit_url`: The full URL to which the final JSON payload should be POSTed.
    5.  `payload`: The JSON payload to be submitted. For "fetch_and_submit" tasks, use the placeholder string "__ANSWER__" for the value that needs to be fetched and processed.

    Example for a "fetch_and_submit" task (e.g., sum numbers from a CSV):
    {{
        "plan": {{
            "task_type": "fetch_and_submit",
            "fetch_url": "https://example.com/data.csv",
            "processing_task": "sum_numbers_in_csv",
            "submit_url": "https://example.com/submit-answer",
            "payload": {{
                "email": "user@example.com",
                "secret": "some_secret",
                "url": "https://example.com/current-task-page",
                "answer": "__ANSWER__"  // This will be replaced by the processed data (e.g., sum)
            }}
        }}
    }}

    Example for a "single_post" task:
    {{
        "plan": {{
            "task_type": "single_post",
            "submit_url": "https://example.com/submit-something",
            "payload": {{
                "email": "user@example.com",
                "answer": "some static value"
            }}
        }}
    }}

    Initial JSON Payload (Your context): --- {initial_payload_str} ---
    Webpage Content (Your instructions): --- {quiz_content} ---
    """
    response = model.generate_content(prompt)
    
    # Return a JSON object containing the result and the model used
    return json.dumps({
        "llm_response": response.text,
        "model_used": model_to_use
    })

if __name__ == "__main__":
    try:
        # Read the combined JSON data from standard input
        input_data = json.load(sys.stdin)
        
        content = input_data["quiz_content"]
        payload = input_data["initial_payload"]
        
        # Run the logic and print the result to standard output
        result_json = run_gemini_logic(content, payload)
        print(result_json)
        
    except Exception as e:
        # Print any errors as a JSON object to stderr
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
