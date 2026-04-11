import os
from pathlib import Path

from google import genai
from google.genai import types


def load_api_key() -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key

    search_roots = [Path.cwd(), Path(__file__).resolve().parent]
    seen = set()

    for root in search_roots:
        for directory in [root, *root.parents]:
            env_path = directory / ".env"
            if env_path in seen or not env_path.exists():
                continue
            seen.add(env_path)

            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue

                key, value = stripped.split("=", 1)
                if key.strip() != "GEMINI_API_KEY":
                    continue

                return value.strip().strip("'\"")

    return None


def main() -> None:
    api_key = load_api_key()
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY in the environment or in a nearby .env file before running this script.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents="Explain why the sky is blue.",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high")
        ),
    )

    print(response.text)


if __name__ == "__main__":
    main()
