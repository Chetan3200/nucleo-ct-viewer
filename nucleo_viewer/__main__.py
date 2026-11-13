import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


def main():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    # One worker: case data lives in memory, not in a database.
    uvicorn.run("nucleo_viewer.app:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
