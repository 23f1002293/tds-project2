import httpx
import asyncio
import os
import json
import random
import string

# --- Configuration ---
# The URL of the main application server
SERVER_URL = "http://127.0.0.1:8000/"
# A valid email and URL to use for successful requests
VALID_EMAIL = "23f1002293@ds.study.iitm.ac.in"
VALID_URL = "https://tds-llm-analysis.s-anand.net/demo"
# The application secret. In a real scenario, this would come from env or config.
# We fetch it from the environment to ensure it matches the server's secret.
APP_SECRET = os.environ.get("APP_SECRET", "test_secret_123")


def generate_random_string(length=10):
    """Generates a random alphanumeric string of a given length."""
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for i in range(length))


async def send_request(client, payload, is_malformed=False):
    """Sends a request to the server and prints the outcome."""
    print("-" * 20)
    headers = {"Content-Type": "application/json"}
    try:
        if is_malformed:
            print(f"Sending MALFORMED JSON: '{payload}'")
            response = await client.post(SERVER_URL, content=payload, headers=headers)
        else:
            print(f"Sending Payload: {json.dumps(payload)}")
            response = await client.post(SERVER_URL, json=payload)

        print(f"Response Status: {response.status_code}")
        # Try to print JSON, but fall back to text if it fails
        try:
            print(f"Response Body: {response.json()}")
        except json.JSONDecodeError:
            print(f"Response Body (non-JSON): {response.text}")
        
        return response.status_code

    except httpx.ConnectError as e:
        print(f"\n[ERROR] Connection failed: {e}")
        print("Please ensure the main application server is running.")
        return None # Indicate connection failure
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        return None


async def main_test_loop():
    """
    Continuously sends various random payloads to the server until a 200 OK
    response is received. This simulates valid and invalid API usage.
    """
    print("--- Starting Test Client ---")
    print(f"Target Server: {SERVER_URL}")
    print(f"Using App Secret: '{APP_SECRET[:4]}...{APP_SECRET[-4:]}'")

    successful = False
    async with httpx.AsyncClient() as client:
        while not successful:
            # Randomly choose which type of request to send
            test_type = random.choice([
                "valid",
                "invalid_secret",
                "missing_fields",
                "malformed_json"
            ])

            status_code = None

            if test_type == "valid":
                print("\nAttempting a VALID request (should result in 200 OK)")
                payload = {
                    "email": VALID_EMAIL,
                    "secret": APP_SECRET,
                    "url": VALID_URL
                }
                status_code = await send_request(client, payload)

            elif test_type == "invalid_secret":
                print("\nAttempting an INVALID SECRET request (should result in 403 Forbidden)")
                payload = {
                    "email": VALID_EMAIL,
                    "secret": f"wrong_{generate_random_string()}",
                    "url": VALID_URL
                }
                status_code = await send_request(client, payload)

            elif test_type == "missing_fields":
                print("\nAttempting a MISSING FIELDS request (should result in 400 Bad Request)")
                payload = {
                    "email": VALID_EMAIL,
                    "secret": APP_SECRET
                    # 'url' key is missing
                }
                if random.choice([True, False]): # Randomly remove email instead
                    payload.pop("email")
                    payload["url"] = VALID_URL

                status_code = await send_request(client, payload)

            elif test_type == "malformed_json":
                print("\nAttempting a MALFORMED JSON request (should result in 400 Bad Request)")
                # Incomplete JSON string
                payload = '{"email": "test@example.com", "secret":'
                status_code = await send_request(client, payload, is_malformed=True)

            if status_code == 200:
                print("\n--- SUCCESS! Received 200 OK. Stopping test client. ---")
                successful = True
            elif status_code is None:
                print("\n--- Connection Error. Stopping test client. ---")
                break # Exit if connection fails

            if not successful:
                # Wait a moment before the next attempt
                await asyncio.sleep(1)

if __name__ == "__main__":
    # Ensure the APP_SECRET is set for the test client to use
    if not os.environ.get("APP_SECRET"):
        print("[WARNING] APP_SECRET environment variable not set.")
        print("Using default 'test_secret_123'. This may cause 403 errors if the server uses a different secret.")
    
    asyncio.run(main_test_loop())