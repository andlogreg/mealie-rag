"""
CLI module.

Contains main entry points.
"""

import logging

import typer
from pythonjsonlogger.json import JsonFormatter

from .config import settings
from .run_fetch import main as fetch_main
from .run_ingest import main as ingest_main
from .run_qa_cli import main as qa_cli_main
from .run_qa_ui import main as qa_ui_main

app = typer.Typer()


def setup_logging():
    formatter = JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(msg)s", style="%"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.dependency_log_level.upper())

    logging.getLogger("mealierag").setLevel(settings.log_level.upper())


@app.callback()
def callback():
    """
    Mealie RAG CLI.
    """
    setup_logging()
    logging.getLogger("mealierag").info(f"Configuration: {settings.model_dump_json()}")


@app.command()
def fetch():
    """
    Fetch all recipes from Mealie. Mostly for debugging purposes.
    """
    fetch_main()


@app.command()
def ingest():
    """Ingest Mealie recipes into the vector database."""
    ingest_main()


@app.command()
def qa_cli():
    """Run the QA Rag interface in the terminal."""
    qa_cli_main()


@app.command()
def qa_ui():
    """Serve the QA Rag UI in the browser."""
    qa_ui_main()


if __name__ == "__main__":
    app()
