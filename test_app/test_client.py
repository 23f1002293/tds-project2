import httpx
import asyncio
import os
import json


async def post_payload():
    url = "http://127.0.0.1:8000/"
    # Use a dummy secret for testing. In a real scenario, this would come from env or config. 
    test_secret = os.environ.get("APP_SECRET", "test_secret_123") 

    payloads = [
        # Valid payload
        {
            "email": "23f1002293@ds.study.iitm.ac.in",
            "secret": test_secret,
            "url": "https://tds-llm-analysis.s-anand.net/demo"
        }
        # # Invalid JSON (missing fields)
        # {
        #     "email": "test@example.com",
        #     "secret": test_secret
        # },
        # # Invalid secret
        # {
        #     "email": "test@example.com",
        #     "secret": "wrong_secret",
        #     "url": "https://tds-llm-analysis.s-anand.net/demo"
        # }
    ]

    invalid_json_payload = "this is not valid json" # Malformed JSON string

    async with httpx.AsyncClient() as client:
        for i, payload in enumerate(payloads):
            print(f"\n--- Sending Payload {i+1} ---")
            print(f"Request URL: {url}")
            print(f"Request Payload: {payload}")
            try:
                response = await client.post(url, json=payload)
                print(f"Response Status Code: {response.status_code}")
                print(f"Response Body: {response.json()}")
            except httpx.ConnectError as e:
                print(f"Error connecting to the server: {e}. Is the main app running?")
            except json.JSONDecodeError as e:
                print(f"--- FAILED TO DECODE JSON RESPONSE ---")
                print(f"Response Status Code: {response.status_code}")
                print(f"Response Content: {response.text}")
                print(f"------------------------------------")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        # # Test case for truly invalid JSON structure
        # print(f"\n--- Sending Invalid JSON Payload (Malformed String) ---")
        # print(f"Request URL: {url}")
        # print(f"Request Payload: '{invalid_json_payload}'")
        # try:
        #     response = await client.post(url, content=invalid_json_payload, headers={"Content-Type": "application/json"})
        #     print(f"Response Status Code: {response.status_code}")
        #     print(f"Response Body: {response.json()}")
        # except httpx.ConnectError as e:
        #     print(f"Error connecting to the server: {e}. Is the main app running?")
        # except Exception as e:
        #     print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(post_payload())
