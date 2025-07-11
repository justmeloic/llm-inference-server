"""
A simple CLI application for chatting with the LLM inference server.
"""

import json

import httpx
import typer
from rich.console import Console
from rich.markdown import Markdown

# --- Configuration ---
SERVER_URL = "http://localhost:8000/api/v1/generate"
APP_NAME = "LLM Server Chat"

# --- Typer App and Console ---
app = typer.Typer(
    name=APP_NAME,
    help="Chat with the LLM inference server from your terminal.",
    add_completion=False,
)
console = Console()


def stream_chat(prompt: str, temperature: float, max_tokens: int):
    """Handles the streaming chat logic."""
    console.print("[bold cyan]Assistant:[/bold cyan]", end=" ")
    full_response = ""
    try:
        with httpx.stream(
            "POST",
            SERVER_URL,
            json={
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
            timeout=None,
        ) as response:
            if response.status_code != 200:
                console.print(
                    f"[bold red]Error: Server returned status {response.status_code}[/bold red]"
                )
                return

            for chunk in response.iter_text():
                if chunk.startswith("data: "):
                    data_str = chunk[len("data: ") :].strip()
                    if data_str == "[DONE]":
                        break
                    if data_str:
                        console.print(f"[cyan]{data_str}[/cyan]", end="")
                        full_response += data_str
    except httpx.ConnectError:
        console.print(
            "[bold red]Connection Error: Could not connect to the server.[/bold red]"
        )
        console.print("Please make sure the LLM inference server is running.")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")

    console.print()  # Newline after response


@app.command()
def chat(
    temperature: float = typer.Option(
        0.7, "--temperature", "-t", help="The temperature for the model's output."
    ),
    max_tokens: int = typer.Option(
        512, "--max-tokens", "-m", help="The maximum number of tokens to generate."
    ),
):
    """
    Starts an interactive chat session with the LLM model.
    """
    console.print(f"[bold green]Welcome to {APP_NAME}![/bold green]")
    console.print("Type 'exit' or 'quit' to end the chat.")
    console.print("-" * 30)

    while True:
        try:
            prompt = console.input("[bold yellow]You: [/bold yellow]")
            if prompt.lower() in ["exit", "quit"]:
                console.print("[bold red]Goodbye![/bold red]")
                break

            stream_chat(prompt, temperature, max_tokens)

        except KeyboardInterrupt:
            console.print("\n[bold red]Goodbye![/bold red]")
            break


def run():
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    run()
