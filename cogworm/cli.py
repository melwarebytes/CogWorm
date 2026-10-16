"""CogWorm CLI — run experiments from the command line.

Usage:
    cogworm run configs/example.yaml
    cogworm run configs/example.yaml --output results/
    cogworm compare results/baseline.json results/defended.json
    cogworm report results/experiment_abc123.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from cogworm.core.graph import AgentNetwork
from cogworm.core.logger import EventLogger
from cogworm.core.models import (
    AgentRole,
    Defense,
    ExperimentConfig,
    SignalType,
    Topology,
)
from cogworm.core.router import Router
from cogworm.core.runner import ExperimentRunner
from cogworm.agents.mock import MockAgent
from cogworm.defenses import DEFENSE_REGISTRY
from cogworm.metrics.calculator import MetricsCalculator

console = Console()


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)

    exp = data.get("experiment", data)

    defenses = []
    for d in exp.get("defenses", []):
        defenses.append(Defense(
            id=d.get("id", d["type"]),
            name=d.get("name", d["type"]),
            config=d.get("config", {}),
        ))

    roles = [AgentRole(r) for r in exp.get("agent_roles", [])]

    return ExperimentConfig(
        name=exp.get("name", "unnamed"),
        topology=Topology(exp.get("topology", "linear")),
        num_agents=exp.get("agents", exp.get("num_agents", 5)),
        **({"agent_roles": roles} if roles else {}),
        model=exp.get("model", "mock"),
        signal_type=SignalType(exp.get("signal_type", "static")),
        injection_agent=exp.get("injection_agent"),
        cycles=exp.get("cycles", 20),
        defenses=defenses,
        seed=exp.get("seed", 42),
    )


def run_experiment(config: ExperimentConfig, output_dir: Path) -> dict:
    """Execute a single experiment and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / f"events_{config.id}.db"

    console.print(f"\n[bold cyan]═══ CogWorm Experiment: {config.name} ═══[/]")
    console.print(f"  Topology:  {config.topology.value}")
    console.print(f"  Agents:    {config.num_agents}")
    console.print(f"  Signal:    {config.signal_type.value}")
    console.print(f"  Cycles:    {config.cycles}")
    console.print(f"  Defenses:  {len(config.defenses)}")
    console.print(f"  Seed:      {config.seed}")
    console.print()

    # Build network
    network = AgentNetwork.from_topology(
        config.topology,
        config.num_agents,
        config.agent_roles or None,
        config.model,
    )

    # Create agent backends
    backends = {}
    signal_content = ""  # needed for LLM fidelity measurement
    from cogworm.signals.factory import create_signal as _cs
    signal_content = _cs(config.signal_type, config.seed)

    for agent_id in network.agent_ids:
        if config.model == "mock":
            backends[agent_id] = MockAgent(seed=config.seed + hash(agent_id))
        elif config.model.startswith("groq:"):
            from cogworm.agents.groq_agent import GroqAgent
            groq_model = config.model.split(":", 1)[1]
            backends[agent_id] = GroqAgent(
                model=groq_model,
                original_signal=signal_content,
            )
            if agent_id == network.agent_ids[0]:
                console.print(f"  [magenta]Model: Groq/{groq_model}[/]")
        elif config.model.startswith("anthropic:"):
            from cogworm.agents.llm_agent import LLMAgent
            model_name = config.model.split(":", 1)[1]
            backends[agent_id] = LLMAgent(
                model=model_name, provider="anthropic",
                original_signal=signal_content,
            )
            if agent_id == network.agent_ids[0]:
                console.print(f"  [magenta]Model: Anthropic/{model_name}[/]")
        elif config.model.startswith("openai:"):
            from cogworm.agents.llm_agent import LLMAgent
            model_name = config.model.split(":", 1)[1]
            backends[agent_id] = LLMAgent(
                model=model_name, provider="openai",
                original_signal=signal_content,
            )
            if agent_id == network.agent_ids[0]:
                console.print(f"  [magenta]Model: OpenAI/{model_name}[/]")
        else:
            # Default: treat as Groq model name for convenience
            from cogworm.agents.groq_agent import GroqAgent
            backends[agent_id] = GroqAgent(
                model=config.model,
                original_signal=signal_content,
            )
            if agent_id == network.agent_ids[0]:
                console.print(f"  [magenta]Model: {config.model}[/]")

    # Set up logger
    logger = EventLogger(db_path)

    # Set up router with defenses
    router = Router()
    for defense_cfg in config.defenses:
        defense_type = defense_cfg.config.get("type", defense_cfg.id)
        if defense_type in DEFENSE_REGISTRY:
            defense_cls = DEFENSE_REGISTRY[defense_type]
            defense = defense_cls(config=defense_cfg.config)
            router.add_defense(defense)
            console.print(f"  [yellow]Defense enabled: {defense.name}[/]")

    # Run
    runner = ExperimentRunner(config, network, backends, logger, router)

    with console.status("[bold green]Running experiment..."):
        result = runner.run()

    # Calculate metrics
    calc = MetricsCalculator(logger)
    metrics = calc.calculate(result)

    # Display results
    _print_results(result, metrics, network)

    # XAI Explanation
    from cogworm.xai.explainer import ExperimentExplainer
    explainer = ExperimentExplainer()
    xai_report = explainer.explain(result, metrics, network, logger)
    console.print(f"\n[bold cyan]═══ XAI Explanation ═══[/]")
    console.print(f"\n[bold]{xai_report.summary}[/]")
    console.print(f"\n[yellow]{xai_report.risk_assessment}[/]")
    if xai_report.critical_path:
        console.print(f"\n[red]Critical Path: {' → '.join(xai_report.critical_path)}[/]")
    for rec in xai_report.recommendations[:3]:
        console.print(f"  [green]→ {rec}[/]")

    # Save XAI report
    xai_text = explainer.format_report(xai_report)
    xai_path = output_dir / f"xai_{config.id}.txt"
    with open(xai_path, "w") as f:
        f.write(xai_text)

    # Save
    result_path = output_dir / f"result_{config.id}.json"
    with open(result_path, "w") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

    events_path = output_dir / f"events_{config.id}.jsonl"
    logger.export_jsonl(events_path)

    # Save metrics
    metrics_path = output_dir / f"metrics_{config.id}.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "attack_success_rate": metrics.attack_success_rate,
            "infection_radius": metrics.infection_radius,
            "propagation_half_life": metrics.propagation_half_life,
            "hop_survival": metrics.hop_survival,
            "mean_fidelity": metrics.mean_fidelity,
            "fidelity_by_hop": metrics.fidelity_by_hop,
            "defense_blocks": metrics.defense_blocks,
            "defense_bypass_rate": metrics.defense_bypass_rate,
            "agents_activated": metrics.agents_activated,
            "agents_contained": metrics.agents_contained,
            "most_common_path": metrics.most_common_path,
            "total_messages": metrics.total_messages,
        }, f, indent=2)

    logger.close()

    console.print(f"\n  [dim]Results:  {result_path}[/]")
    console.print(f"  [dim]Events:   {events_path}[/]")
    console.print(f"  [dim]Metrics:  {metrics_path}[/]")
    console.print(f"  [dim]DB:       {db_path}[/]")

    return {
        "result": result.model_dump(mode="json"),
        "metrics_path": str(metrics_path),
    }


def _print_results(result, metrics, network) -> None:
    """Pretty-print experiment results."""
    table = Table(title="Experiment Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Propagation Rate", f"{result.propagation_rate:.1%}")
    table.add_row("Agents Activated", f"{result.agents_activated}/{result.config.num_agents}")
    table.add_row("Agents Contained", f"{result.agents_contained}")
    table.add_row("Infection Radius", f"{result.infection_radius} hops")
    table.add_row("Hop Survival", f"{metrics.hop_survival} hops")
    table.add_row("Half-Life", f"{metrics.propagation_half_life} hops")
    table.add_row("Mean Fidelity", f"{result.mean_signal_fidelity:.3f}")
    table.add_row("Defense Blocks", f"{result.defense_blocks}")
    table.add_row("Total Messages", f"{result.total_messages}")
    table.add_row("Duration", f"{result.duration_seconds:.3f}s")

    if metrics.most_common_path:
        path_str = " → ".join(metrics.most_common_path)
        table.add_row("Propagation Path", path_str)

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CogWorm — Adversarial Propagation Benchmark"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("config", help="Path to experiment config YAML")
    run_parser.add_argument(
        "--output", "-o", default="results",
        help="Output directory (default: results/)",
    )

    # Compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare two results")
    cmp_parser.add_argument("baseline", help="Baseline result JSON")
    cmp_parser.add_argument("defended", help="Defended result JSON")

    args = parser.parse_args()

    if args.command == "run":
        config = load_config(args.config)
        run_experiment(config, Path(args.output))
    elif args.command == "compare":
        console.print("[yellow]Compare mode — coming soon[/]")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
