FROM python:3.11-slim

# 1. Install necessary system dependencies (Run as root is fine here)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libnss3 libatk-bridge2.0-0 libxss1 libgbm1 libgtk-3-0 libasound2 libnspr4 \
        libatk1.0-0 libcups2 libdbus-1-3 libdrm2 libexpat1 libglib2.0-0 libnss3 \
        libpango-1.0-0 libpangocairo-1.0-0 libx11-6 libxcomposite1 libxdamage1 \
        libxext6 libxfixes3 libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Set the environment variable for the browser installation path
# We use /usr/local/bin for maximum visibility and compatibility.
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/local/bin/ms-playwright

# 3. Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Install the browser binaries to the new, explicit location
# This must run after the system dependencies and Python packages are installed.
RUN python -m playwright install --with-deps

# 5. Copy the rest of your application code
COPY . .

# 6. Configure the application to run as the non-root user (cnb)
# This is best practice for Cloud Run security and avoids root-related issues.
# Cloud Run automatically creates the cnb user.
USER cnb

# 7. Define the environment and command
ENV PORT 8080
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]