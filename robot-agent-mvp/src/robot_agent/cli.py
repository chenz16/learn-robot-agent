"""CLI entry point for Robot Agent MVP."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
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

# Preset demo tasks
DEMO_TASKS = {
    "/demo1": {
        "name": "Task A: Repeat Decomposition",
        "instruction": "put both the alphabet soup and the tomato sauce in the basket",
        "objects": {
            "alphabet_soup": [0.2, -0.1, 0.10],
            "tomato_sauce": [-0.1, 0.15, 0.10],
            "basket": [0.6, 0.0, 0.05],
        },
    },
    "/demo2": {
        "name": "Task B: Chain Decomposition",
        "instruction": "put the black bowl in the bottom drawer of the cabinet and close it",
        "objects": {
            "black_bowl": [0.2, 0.1, 0.10],
            "bottom_drawer": [0.6, -0.2, 0.15],
            "cabinet": [0.6, -0.2, 0.30],
        },
    },
    "/demo3": {
        "name": "Simple: Pick and Place",
        "instruction": "pick up the red cup and place it on the plate",
        "objects": {
            "red_cup": [0.15, 0.0, 0.08],
            "plate": [0.5, 0.1, 0.02],
        },
    },
}


# ---------------------------------------------------------------------------
# Environment state display
# ---------------------------------------------------------------------------

def _snapshot_env(env) -> dict[str, Any]:
    """Capture a serializable snapshot of environment state."""
    obs = env.get_observation()
    return {
        "ee_pos": list(obs["robot0_eef_pos"]),
        "gripper": "open" if obs["robot0_gripper_qpos"][0] > 0.5 else "closed",
        "holding": obs.get("holding"),
        "objects": {k: list(v) for k, v in obs.get("objects", {}).items()},
    }


def _env_changed(before: dict, after: dict) -> bool:
    """Check if environment state changed between snapshots."""
    return before != after


def _show_env_table(state: dict, title: str) -> None:
    """Display environment state as a rich table."""
    table = Table(title=title, show_lines=True, title_style="bold")
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Value", style="white")

    ee = state["ee_pos"]
    table.add_row("End-Effector", f"[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")
    table.add_row("Gripper", state["gripper"])
    table.add_row("Holding", state["holding"] or "-")

    for name, pos in state["objects"].items():
        table.add_row(f"  {name}", f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    console.print(table)


def _show_env_diff(before: dict, after: dict) -> None:
    """Display BEFORE/AFTER environment state side by side when state changed."""
    console.print()
    console.rule("[bold yellow]Simulation State Change[/bold yellow]")

    table = Table(show_lines=True, title_style="bold")
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Before", style="dim")
    table.add_column("After", style="green")

    # EE position
    be, ae = before["ee_pos"], after["ee_pos"]
    b_str = f"[{be[0]:.3f}, {be[1]:.3f}, {be[2]:.3f}]"
    a_str = f"[{ae[0]:.3f}, {ae[1]:.3f}, {ae[2]:.3f}]"
    style = "bold green" if be != ae else ""
    table.add_row("End-Effector", b_str, Text(a_str, style=style))

    # Gripper
    style = "bold green" if before["gripper"] != after["gripper"] else ""
    table.add_row("Gripper", before["gripper"], Text(after["gripper"], style=style))

    # Holding
    bh = before["holding"] or "-"
    ah = after["holding"] or "-"
    style = "bold green" if bh != ah else ""
    table.add_row("Holding", bh, Text(ah, style=style))

    # Objects
    all_names = sorted(set(list(before["objects"]) + list(after["objects"])))
    for name in all_names:
        bp = before["objects"].get(name)
        ap = after["objects"].get(name)
        b_str = f"[{bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f}]" if bp else "-"
        a_str = f"[{ap[0]:.3f}, {ap[1]:.3f}, {ap[2]:.3f}]" if ap else "-"
        changed = bp != ap
        style = "bold green" if changed else ""
        table.add_row(f"  {name}", b_str, Text(a_str, style=style))

    console.print(table)
    console.print()


def _show_demo_menu() -> None:
    """Display available demo tasks."""
    console.print()
    table = Table(title="Preset Demo Tasks", show_lines=True)
    table.add_column("Command", style="bold cyan", width=8)
    table.add_column("Task", style="white")
    table.add_column("Instruction", style="yellow")

    for cmd, task in DEMO_TASKS.items():
        table.add_row(cmd, task["name"], task["instruction"])

    console.print(table)
    console.print("[dim]Type a command above, or type any instruction directly.[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# Config & provider
# ---------------------------------------------------------------------------

def _resolve_config_path(config_flag: str | None) -> Path:
    if config_flag:
        return Path(config_flag)
    return Path(__file__).parent.parent.parent / "config" / "agent.json"


def _resolve_workspace_path(config_path: Path) -> Path:
    return config_path.parent.parent / "workspace"


def _load_env(config_path: Path) -> None:
    """Load .env file from project root (if exists)."""
    env_file = config_path.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _make_provider(config_path: Path):
    """Create LLM provider from agent config."""
    _load_env(config_path)

    if not config_path.exists():
        console.print(f"[yellow]Config not found at {config_path}, using defaults[/yellow]")

    with open(config_path) as f:
        raw = json.load(f) if config_path.exists() else {}

    agents = raw.get("agents", {}).get("defaults", {})
    model = agents.get("model", "gemini/gemini-2.0-flash")
    provider_name = agents.get("provider", "gemini")

    # LiteLLM providers: gemini, openai, anthropic, etc.
    if provider_name != "custom":
        from nanobot.providers.litellm_provider import LiteLLMProvider
        providers = raw.get("providers", {})
        p = providers.get(provider_name, {})
        api_key = p.get("apiKey") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return LiteLLMProvider(
            api_key=api_key,
            api_base=p.get("apiBase"),
            default_model=model,
        )

    # Custom: direct OpenAI-compatible endpoint
    from nanobot.providers.custom_provider import CustomProvider
    providers = raw.get("providers", {})
    custom = providers.get("custom", {})
    return CustomProvider(
        api_key=custom.get("apiKey", "no-key"),
        api_base=custom.get("apiBase", "http://localhost:8000/v1"),
        default_model=model,
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


def _print_response(response: str, markdown: bool) -> None:
    body = Markdown(response) if markdown else Text(response)
    console.print()
    console.print("[cyan]robot-agent[/cyan]")
    console.print(body)
    console.print()


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------

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
    from robot_agent.env.mock import MockEnv
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

    console.print(Panel(
        f"[green]Environment:[/green] {env_type}  |  "
        f"[green]VLA:[/green] {vla_type}  |  "
        f"[green]Model:[/green] {defaults['model'] or 'default'}",
        title="[bold cyan]Robot Agent MVP[/bold cyan]",
        expand=False,
    ))

    if message:
        # Single message mode
        async def run_once():
            env.reset()
            before = _snapshot_env(env)

            async def _progress(content: str, *, tool_hint: bool = False) -> None:
                if tool_hint:
                    console.print(f"  [magenta]{content}[/magenta]")
                else:
                    console.print(f"  [dim]{content}[/dim]")

            response = await agent_loop.process_direct(message, on_progress=_progress)

            after = _snapshot_env(env)
            if _env_changed(before, after):
                _show_env_diff(before, after)

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

        env.reset()
        console.print()
        console.print("Type [bold]/demo[/bold] to see preset tasks, or type any instruction.")
        console.print("Type [bold]exit[/bold] or Ctrl+C to quit.\n")

        def _exit_on_sigint(signum, frame):
            console.print("\nGoodbye!")
            os._exit(0)

        signal.signal(signal.SIGINT, _exit_on_sigint)

        async def run_interactive():
            bus_task = asyncio.create_task(agent_loop.run())
            turn_done = asyncio.Event()
            turn_done.set()
            turn_response: list[str] = []
            robot_tools_used: list[str] = []

            ROBOT_TOOL_NAMES = {
                "look", "move", "grasp", "perceive",
                "start_subtask", "check_loops", "wait_subtask",
                "emergency_stop", "model_health", "model_ensure",
            }

            async def _consume_outbound():
                while True:
                    try:
                        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                        if msg.metadata.get("_progress"):
                            content = msg.content
                            is_tool = msg.metadata.get("_tool_hint", False)
                            if is_tool:
                                # Track robot tool usage
                                for tn in ROBOT_TOOL_NAMES:
                                    if tn in content:
                                        robot_tools_used.append(tn)
                                        break
                                console.print(f"  [magenta]{content}[/magenta]")
                            else:
                                console.print(f"  [dim]{content}[/dim]")
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

                        # Handle /demo menu
                        if command.lower() == "/demo":
                            _show_demo_menu()
                            continue

                        # Handle /demo1, /demo2, /demo3 — reset env with task objects
                        actual_input = command
                        if command.lower() in DEMO_TASKS:
                            demo = DEMO_TASKS[command.lower()]
                            if isinstance(env, MockEnv):
                                env._objects = {k: list(v) for k, v in demo["objects"].items()}
                                env.reset()
                            console.print(Panel(
                                f"[bold]{demo['name']}[/bold]\n[yellow]{demo['instruction']}[/yellow]",
                                expand=False,
                            ))
                            actual_input = demo["instruction"]

                        # Snapshot env before
                        before = _snapshot_env(env)

                        turn_done.clear()
                        turn_response.clear()
                        robot_tools_used.clear()

                        await bus.publish_inbound(InboundMessage(
                            channel="cli", sender_id="user", chat_id="direct",
                            content=actual_input,
                        ))

                        with console.status("[dim]thinking...[/dim]", spinner="dots"):
                            await turn_done.wait()

                        # Snapshot env after
                        after = _snapshot_env(env)

                        # If env changed → show simulation diff
                        if _env_changed(before, after):
                            _show_env_diff(before, after)

                        # Show agent response
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
