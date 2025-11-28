import os
import json
import sys
import re
import google.generativeai as genai

# This script can perform two tasks:
# 1. "generate_plan": Reads webpage content and creates a plan for the main app.
# 2. "extract_answer": Reads content and a question, and extracts the answer.

def get_gemini_model():
    """Initializes and returns the Gemini model."""
    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=gemini_api_key)

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
    
    return genai.GenerativeModel(model_to_use)

def generate_plan(quiz_content: str, initial_payload: dict):
    """Generates a JSON plan based on the webpage content."""
    model = get_gemini_model()
    initial_payload_str = json.dumps(initial_payload, indent=2)
    prompt = f"""
    You are an intelligent agent. Your task is to analyze the webpage content below and create a JSON plan to solve the described task.
    You must use the provided "Initial JSON Payload" to fill in any required data.

    Analyze the task and respond ONLY with a JSON object with a single key "plan".
    The "plan" object should contain:
    1.  `task_type`: A string describing the task ("fetch_and_submit" or "single_post").
    2.  `fetch_url`: (For "fetch_and_submit") The full URL to fetch data from. 
        - CRITICAL: If you see an <audio> or <video> tag, this URL MUST be the value of its `src` attribute. Resolve it to a full URL if it's relative.
        - If you see a link to a data file (.csv, .json, etc.), use that URL.
        - IMPORTANT: Do not assume a directory structure. If a URL looks like `https://example.com/data?id=123`, the data is at `https://example.com/data?id=123`, not `https://example.com/data/index.csv?id=123`.
    3.  `processing_task`: (Optional) E.g., "sum_numbers_in_csv", "extract_secret_code", "transcribe_audio".
        - If the `fetch_url` points to an audio or video file, set this to "transcribe_audio".
    4.  `submit_url`: The full URL to POST the final payload to.
    5.  `payload`: The JSON payload to be submitted, using "__ANSWER__" as a placeholder for processed data.

    Initial JSON Payload (Your context): --- {initial_payload_str} ---
    Webpage Content (Your instructions): --- {quiz_content} ---
    """
    response = model.generate_content(prompt)
    return json.dumps({"llm_response": response.text})

def extract_answer(content: str, question: str):
    """Extracts a specific answer from content based on a question."""
    model = get_gemini_model()
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
    response = model.generate_content(prompt)
    return json.dumps({"answer": response.text.strip()})

def process_data(data_content: str, processing_task: str):
    """Processes raw data using the LLM based on a specific task."""
    model = get_gemini_model()
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
    response = model.generate_content(prompt)
    return json.dumps({"processed_data": response.text.strip()})

if __name__ == "__main__":
    try:
        input_data = json.load(sys.stdin)
        task = input_data.get("task")

        if task == "generate_plan":
            result_json = generate_plan(input_data["quiz_content"], input_data["initial_payload"])
            print(result_json)
        
        elif task == "extract_answer":
            result_json = extract_answer(input_data["content"], input_data["question"])
            print(result_json)
            
        elif task == "process_data":
            result_json = process_data(input_data["data_content"], input_data["processing_task"])
            print(result_json)

        else:
            raise ValueError(f"Unknown task: {task}")
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
