# Gemini API Project

This project uses the Gemini API to perform various tasks, including natural language processing and multimodal analysis.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set environment variables:**
    Create a `.env` file and add the following:
    ```
    GEMINI_API_KEY="YOUR_API_KEY"
    APP_SECRET="YOUR_APP_SECRET"
    ```

## Running the Application

```bash
python app.py
```

## Changes

- Fixed a `ValueError` that occurred when the Gemini API returned an empty response. Added safety checks to handle these cases gracefully.
- Removed the unused `temp_file.csv`.
