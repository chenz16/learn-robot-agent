"""CLI entry point for Robot Agent MVP."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

app = typer.Typer(
    name="robot-agent",
    help="Robot Agent MVP: LLM task decomposition + VLA execution",
    no_args_is_help=True,
)

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}


def _print_response(response: str, markdown: bool) -> None:
    body = Markdown(response) if markdown else Text(response)
    console.print()
    console.print("[cyan]robot-agent[/cyan]")
    console.print(body)
    console.print()


def _resolve_config_path(config_flag: str | None) -> Path:
    if config_flag:
        return Path(config_flag)
    # Default: config/agent.json relative to this package
    return Path(__file__).parent.parent.parent / "config" / "agent.json"


def _resolve_workspace_path(config_path: Path) -> Path:
    # Default: workspace/ relative to project root
    return config_path.parent.parent / "workspace"


def _make_provider(config_path: Path):
    """Create LLM provider from agent config."""
    from nanobot.providers.custom_provider import CustomProvider

    if config_path.exists():
        with open(config_path) as f:
            raw = json.load(f)
        providers = raw.get("providers", {})
        custom = providers.get("custom", {})
        agents = raw.get("agents", {}).get("defaults", {})
        return CustomProvider(
            api_key=custom.get("apiKey", "no-key"),
            api_base=custom.get("apiBase", "http://localhost:8000/v1"),
            default_model=agents.get("model", "qwen2.5-72b"),
        )
    else:
        console.print(f"[yellow]Config not found at {config_path}, using defaults[/yellow]")
        return CustomProvider(
            api_key="no-key",
            api_base="http://localhost:8000/v1",
            default_model="qwen2.5-72b",
        )


def _get_agent_defaults(config_path: Path) -> dict:
    """Extract agent defaults from config."""
    defaults = {
        "model": None,
        "temperature": 0.1,
        "max_tokens": 8192,
        "max_iterations": 40,
        "memory_window": 50,
    }
    if config_path.exists():
        with open(config_path) as f:
            raw = json.load(f)
        cfg = raw.get("agents", {}).get("defaults", {})
        if cfg.get("model"):
            defaults["model"] = cfg["model"]
        if cfg.get("temperature") is not None:
            defaults["temperature"] = cfg["temperature"]
        if cfg.get("maxTokens") is not None:
            defaults["max_tokens"] = cfg["maxTokens"]
        if cfg.get("maxToolIterations") is not None:
            defaults["max_iterations"] = cfg["maxToolIterations"]
        if cfg.get("memoryWindow") is not None:
            defaults["memory_window"] = cfg["memoryWindow"]
    return defaults


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Single message to send"),
    env_type: str = typer.Option("mock", "--env", help="Environment: mock or libero"),
    vla_type: str = typer.Option("mock", "--vla", help="VLA adapter: mock or http"),
    vla_url: str = typer.Option("http://localhost:8020", "--vla-url", help="VLA HTTP endpoint"),
    vlm_url: str = typer.Option(None, "--vlm-url", help="VLM endpoint (optional)"),
    config: str = typer.Option(None, "--config", "-c", help="Path to agent.json"),
    task_name: str = typer.Option(None, "--task", help="LIBERO task name (for libero env)"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render as Markdown"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show runtime logs"),
):
    """Interact with the robot agent."""
    from loguru import logger
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    from robot_agent.context import RobotContext
    from robot_agent.loop import LoopManager
    from robot_agent.routing import RouteConfig
    from robot_agent.safety import SafetyManager
    from robot_agent.tools import register_robot_tools

    if not logs:
        logger.disable("nanobot")
        logger.disable("robot_agent")

    config_path = _resolve_config_path(config)
    workspace = _resolve_workspace_path(config_path)
    workspace.mkdir(parents=True, exist_ok=True)

    # Load route config
    routes_path = config_path.parent / "routes.yaml"
    route_config = RouteConfig.load(routes_path) if routes_path.exists() else RouteConfig.default()

    # Create environment
    if env_type == "libero":
        from robot_agent.env.libero import LiberoEnv
        env = LiberoEnv(task_name=task_name or "libero_10:0")
    else:
        from robot_agent.env.mock import MockEnv
        env = MockEnv()

    # Create VLA adapter
    if vla_type == "http":
        from robot_agent.vla.http import HTTPVLAAdapter
        vla = HTTPVLAAdapter(base_url=vla_url)
    else:
        from robot_agent.vla.mock import MockVLAAdapter
        vla = MockVLAAdapter(action_dim=env.action_dim)

    # Create shared components
    safety = SafetyManager()
    loop_manager = LoopManager()
    ctx = RobotContext(
        env=env, vla=vla, loop_manager=loop_manager,
        safety=safety, route_config=route_config, vlm_url=vlm_url,
    )

    # Create nanobot agent loop
    provider = _make_provider(config_path)
    bus = MessageBus()
    defaults = _get_agent_defaults(config_path)

    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=workspace,
        model=defaults["model"],
        temperature=defaults["temperature"],
        max_tokens=defaults["max_tokens"],
        max_iterations=defaults["max_iterations"],
        memory_window=defaults["memory_window"],
    )

    # Register robot tools
    register_robot_tools(agent_loop.tools, ctx)

    console.print(f"[green]Environment:[/green] {env_type}")
    console.print(f"[green]VLA adapter:[/green] {vla_type}")
    console.print(f"[green]Tools:[/green] {', '.join(agent_loop.tools.tool_names)}")
    console.print()

    if message:
        # Single message mode
        async def run_once():
            async def _progress(content: str, *, tool_hint: bool = False) -> None:
                console.print(f"  [dim]{content}[/dim]")

            response = await agent_loop.process_direct(message, on_progress=_progress)
            _print_response(response, markdown)
            await agent_loop.close_mcp()

        asyncio.run(run_once())
    else:
        # Interactive mode
        from nanobot.bus.events import InboundMessage

        history_dir = workspace / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        prompt_session = PromptSession(
            history=FileHistory(str(history_dir / "cli_history")),
            multiline=False,
        )

        console.print("[cyan]robot-agent[/cyan] interactive mode (type [bold]exit[/bold] or Ctrl+C to quit)\n")

        def _exit_on_sigint(signum, frame):
            console.print("\nGoodbye!")
            os._exit(0)

        signal.signal(signal.SIGINT, _exit_on_sigint)

        async def run_interactive():
            bus_task = asyncio.create_task(agent_loop.run())
            turn_done = asyncio.Event()
            turn_done.set()
            turn_response: list[str] = []

            async def _consume_outbound():
                while True:
                    try:
                        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                        if msg.metadata.get("_progress"):
                            console.print(f"  [dim]{msg.content}[/dim]")
                        elif not turn_done.is_set():
                            if msg.content:
                                turn_response.append(msg.content)
                            turn_done.set()
                        elif msg.content:
                            _print_response(msg.content, markdown)
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

            outbound_task = asyncio.create_task(_consume_outbound())

            try:
                while True:
                    try:
                        with patch_stdout():
                            user_input = await prompt_session.prompt_async(
                                HTML("<b fg='ansigreen'>You:</b> ")
                            )
                        command = user_input.strip()
                        if not command:
                            continue
                        if command.lower() in EXIT_COMMANDS:
                            console.print("\nGoodbye!")
                            break

                        turn_done.clear()
                        turn_response.clear()

                        await bus.publish_inbound(InboundMessage(
                            channel="cli", sender_id="user", chat_id="direct",
                            content=user_input,
                        ))

                        with console.status("[dim]thinking...[/dim]", spinner="dots"):
                            await turn_done.wait()

                        if turn_response:
                            _print_response(turn_response[0], markdown)
                    except (KeyboardInterrupt, EOFError):
                        console.print("\nGoodbye!")
                        break
            finally:
                agent_loop.stop()
                outbound_task.cancel()
                await asyncio.gather(bus_task, outbound_task, return_exceptions=True)
                await agent_loop.close_mcp()

        asyncio.run(run_interactive())


if __name__ == "__main__":
    app()
