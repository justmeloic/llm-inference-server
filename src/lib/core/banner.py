"""
This module provides a cool, informative banner for the server startup.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import Settings


def print_banner(console: Console, settings: Settings):
    """
    Prints a banner to the console.
    """

    # ASCII Art for the title
    title_art = r"""
██╗     ██╗     ███╗   ███╗███████╗███████╗██████╗ ██╗   ██╗███████╗██╗   ██╗
██║     ██║     ████╗ ████║██╔════╝██╔════╝██╔══██╗██║   ██║██╔════╝██║   ██║
██║     ██║     ██╔████╔██║███████╗█████╗  ██████╔╝██║   ██║█████╗  ██║   ██║
██║     ██║     ██║╚██╔╝██║╚════██║██╔══╝  ██╔══██╗██║   ██║██╔══╝  ╚██╗ ██╔╝
███████╗███████╗██║ ╚═╝ ██║███████║███████╗██║  ██║╚██████╔╝███████╗ ╚████╔╝
╚══════╝╚══════╝╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝  ╚═══╝
    """

    console.print(Text(title_art, style="bold blue"), justify="center")

    # Configuration Table
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="right", style="bold cyan")
    grid.add_column(justify="left", style="white")

    grid.add_row("Model Path :", f"[italic]{settings.model_path}[/italic]")
    grid.add_row("GPU Layers :", f"[green]{settings.n_gpu_layers}[/green]")
    grid.add_row("Context Size :", f"[yellow]{settings.n_ctx}[/yellow]")
    grid.add_row("Batch Size :", f"[magenta]{settings.max_batch_size}[/magenta]")
    grid.add_row(
        "Server URL :",
        f"[link=http://{settings.server_host}:{settings.server_port}]http://{settings.server_host}:{settings.server_port}[/link]",
    )

    panel = Panel(
        grid,
        title="[bold]Server Configuration[/bold]",
        border_style="blue",
        expand=False,
        padding=(1, 2),
    )

    console.print(panel)
