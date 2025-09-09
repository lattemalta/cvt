from pathlib import Path

from rich.traceback import install
from typer import Typer

from .logger import setup_logger

install(show_locals=True)
ROOT = Path(__file__).parents[2].resolve()


app = Typer()


@app.command(no_args_is_help=True)
def main(verbose: bool = False) -> None:
    setup_logger(verbose)
