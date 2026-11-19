import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors


def main() -> int:
    # Resolve configuration relative to the project, not the terminal location.
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env", encoding="utf-8-sig")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    if not api_key or api_key == "replace_with_your_api_key":
        print("Configuration error: set GEMINI_API_KEY in your .env file.")
        return 1

    if not model:
        print("Configuration error: GEMINI_MODEL cannot be empty.")
        return 1

    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model=model,
                contents="Reply with exactly: PolicyPilot connection successful",
            )

        if not response.text:
            print("The request completed, but no text was returned.")
            return 1

        print(f"Model requested: {model}")
        print(f"Response: {response.text.strip()}")
        return 0

    except errors.APIError as exc:
        print(f"Gemini API request failed. Status code: {exc.code}")
        print("Check the key, model access, quota, and billing configuration.")
        return 1

    except Exception as exc:
        print(f"Connection test failed: {type(exc).__name__}")
        print("Check package installation, network access, and configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
