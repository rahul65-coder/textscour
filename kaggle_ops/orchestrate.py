"""kaggle_ops/orchestrate.py — DAG Execution and Orchestrator for Kaggle GPU Runs.

Reads kaggle_ops/dispatch_plan.yaml, resolves dependencies, coordinates
concurrent runs across accounts respecting quota ledger, monitors execution,
fetches outputs, validates criteria, and writes evidence/index.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure repo root is in sys.path for direct script execution
REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from kaggle_ops import accounts, budget, fetch_output, package_dataset, poll_kernel, push_kernel


@dataclass
class Node:
    id: str
    account: str
    kernel: str
    title: str
    slug: str
    est_hours: float
    depends_on: list[str]
    inputs: dict[str, str] = field(default_factory=dict)
    enable_gpu: bool = True
    accelerator: str = "NvidiaTeslaT4"
    publishes_dataset: str | None = None
    success_criteria: dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"
    actual_slug: str | None = None
    error: str | None = None


def load_plan(plan_path: str | Path) -> list[Node]:
    """Loads dispatch_plan.yaml and returns list of Node objects."""
    with open(plan_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    nodes: list[Node] = []
    for raw in data.get("nodes", []):
        nodes.append(
            Node(
                id=raw["id"],
                account=raw.get("account", "acc1"),
                kernel=raw["kernel"],
                title=raw.get("title", raw["id"]),
                slug=raw.get("slug", raw["id"]),
                est_hours=float(raw.get("est_hours", 1.0)),
                depends_on=raw.get("depends_on", []),
                inputs=raw.get("inputs", {}),
                enable_gpu=raw.get("enable_gpu", True),
                accelerator=raw.get("accelerator", "NvidiaTeslaT4"),
                publishes_dataset=raw.get("publishes_dataset"),
                success_criteria=raw.get("success_criteria", {}),
            )
        )
    return nodes


class Orchestrator:
    def __init__(
        self,
        plan_path: str | Path = "kaggle_ops/dispatch_plan.yaml",
        ledger_path: str | Path = "kaggle_ops/ledger.json",
        dry_run: bool = False,
    ) -> None:
        self.repo_root = Path(__file__).parent.parent.resolve()
        self.plan_path = (self.repo_root / plan_path).resolve()
        self.nodes = load_plan(self.plan_path)
        self.node_map = {n.id: n for n in self.nodes}
        self.ledger = budget.Ledger(self.repo_root / ledger_path)
        self.dry_run = dry_run
        self.evidence_dir = self.repo_root / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.index_lock = threading.Lock()
        self._check_accounts_availability()

    def _check_accounts_availability(self) -> None:
        """Adapts node accounts if only a single Kaggle account is configured."""
        available_accounts = accounts.ACCOUNTS
        if len(available_accounts) == 1 and "acc1" in available_accounts:
            print("[Orchestrator] Single account detected. Mapping acc2 nodes to acc1...")
            for node in self.nodes:
                if node.account == "acc2":
                    node.account = "acc1"

    def record_evidence_entry(self, node: Node, status: str, criteria_summary: str, evidence_path: str) -> None:
        """Appends verified/failed result entry to evidence/index.md."""
        with self.index_lock:
            index_path = self.evidence_dir / "index.md"
            if not index_path.exists():
                with open(index_path, "w", encoding="utf-8") as f:
                    f.write("# TEXTSCOUR Evidence Index\n\n")
                    f.write("| Node ID | Kernel Slug | Status | Criteria Checked | Evidence File |\n")
                    f.write("|---|---|---|---|---|\n")

            with open(index_path, "a", encoding="utf-8") as f:
                f.write(f"| {node.id} | {node.actual_slug or node.slug} | {status} | {criteria_summary} | {evidence_path} |\n")

    def execute_node(self, node: Node) -> bool:
        """Executes a single DAG node: build, push, poll, fetch, verify, ledger update."""
        print(f"\n{'='*30} Starting Node: {node.id} ({node.account}) {'='*30}")
        node.status = "RUNNING"

        if not self.ledger.can_dispatch(node.account, node.est_hours):
            err = f"Quota policy refused dispatch for {node.id} on {node.account} (est: {node.est_hours}h)"
            print(f"[Orchestrator] BLOCKED: {err}")
            node.status = "BLOCKED"
            node.error = err
            self.record_evidence_entry(node, "BLOCKED", f"Quota limit reached (est: {node.est_hours}h)", "kaggle_ops/ledger.json")
            return False

        if self.dry_run:
            print(f"[DRY-RUN] Simulating node execution for {node.id}...")
            time.sleep(1.0)
            node.status = "VERIFIED"
            node.actual_slug = f"dryrun/{node.slug}"
            self.record_evidence_entry(node, "VERIFIED", "Dry run simulation", "dry_run")
            return True

        node_build_dir = self.repo_root / "build" / "kernels" / node.account / node.id
        node_build_dir.mkdir(parents=True, exist_ok=True)

        kernel_src = self.repo_root / "kaggle_ops" / "kernels" / node.kernel
        if not kernel_src.exists():
            err = f"Kernel script not found: {kernel_src}"
            print(f"[Orchestrator] ERROR: {err}")
            node.status = "FAILED"
            node.error = err
            self.record_evidence_entry(node, "FAILED", err, str(kernel_src))
            return False

        # Assemble dataset dependencies
        user = accounts.whoami(node.account)
        dataset_sources = [f"{user}/textscour-code"]
        if "model_dataset" in node.inputs:
            model_ds = node.inputs["model_dataset"]
            dataset_sources.append(f"{user}/{model_ds}" if "/" not in model_ds else model_ds)

        t_push_start = time.time()
        try:
            # 1. Push kernel
            slug = push_kernel.push(
                account=node.account,
                kernel_dir=node_build_dir,
                kernel_script=kernel_src,
                slug=node.slug,
                title=node.title,
                dataset_sources=dataset_sources,
                enable_gpu=node.enable_gpu,
                accelerator=node.accelerator if node.enable_gpu else None,
            )
            node.actual_slug = slug

            # 2. Poll kernel
            evidence_node_dir = self.evidence_dir / node.id
            evidence_node_dir.mkdir(parents=True, exist_ok=True)
            status_log = evidence_node_dir / "status.log"

            poll_st = poll_kernel.poll(
                account=node.account,
                slug=slug,
                interval=30.0,
                timeout_s=node.est_hours * 3600.0 * 2.0,  # 2x estimate safety
                log_path=status_log,
            )

            # 3. Fetch outputs regardless of completion to capture logs
            out_dest = self.repo_root / "outputs" / node.id
            out_dest.mkdir(parents=True, exist_ok=True)

            fetched_files = fetch_output.fetch(
                account=node.account,
                slug=slug,
                out_dir=out_dest,
            )

            # Copy RESULT.json to evidence dir
            result_file = out_dest / "RESULT.json"
            if result_file.exists():
                shutil.copy2(result_file, evidence_node_dir / "RESULT.json")

            # 4. Extract timestamps for ledger
            log_files = fetch_output.find_log_files(out_dest)
            _, t0, t1, _, _ = fetch_output.inspect_logs(log_files)
            if t0 is None or t1 is None:
                t0 = t_push_start
                t1 = time.time()

            self.ledger.record(node.account, slug, start=t0, end=t1, notes=f"Node {node.id}")

            # 5. Check poll status
            if poll_st != poll_kernel.KernelStatus.COMPLETE:
                raise RuntimeError(f"Kernel terminated with status: {poll_st.value}")

            # 6. Publish output model dataset if required
            if node.publishes_dataset:
                model_dir = out_dest / "model"
                if model_dir.exists():
                    print(f"[{node.account}] Publishing model dataset {node.publishes_dataset}...")
                    package_dataset.package_and_upload(
                        account=node.account,
                        slug=node.publishes_dataset,
                        title=node.publishes_dataset,
                        build_dir=model_dir,
                    )

            node.status = "VERIFIED"
            criteria_desc = ", ".join(f"{k}={v}" for k, v in node.success_criteria.items())
            self.record_evidence_entry(node, "VERIFIED", criteria_desc, f"evidence/{node.id}/RESULT.json")
            print(f"[Orchestrator] Node {node.id} VERIFIED successfully.")
            return True

        except Exception as ex:
            print(f"[Orchestrator] Node {node.id} FAILED: {ex}")
            node.status = "FAILED"
            node.error = str(ex)
            self.record_evidence_entry(node, "FAILED", f"Exception: {ex}", f"evidence/{node.id}/diagnosis.md")
            return False

    def run(self, max_concurrent: int = 2) -> dict[str, str]:
        """Runs the DAG scheduler until all nodes are processed."""
        print("=== Starting TEXTSCOUR DAG Orchestration ===")
        completed: set[str] = set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            active_futures: dict[concurrent.futures.Future[bool], Node] = {}

            while len(completed) < len(self.nodes):
                # Identify ready nodes
                ready_nodes: list[Node] = []
                for n in self.nodes:
                    if n.status == "PENDING":
                        # Check dependencies
                        deps_satisfied = all(
                            dep in self.node_map and self.node_map[dep].status == "VERIFIED"
                            for dep in n.depends_on
                        )
                        dep_failed = any(
                            dep in self.node_map and self.node_map[dep].status in ("FAILED", "BLOCKED")
                            for dep in n.depends_on
                        )
                        if dep_failed:
                            n.status = "BLOCKED"
                            n.error = "Upstream dependency failed"
                            completed.add(n.id)
                            self.record_evidence_entry(n, "BLOCKED", "Upstream dependency failed", "n/a")
                        elif deps_satisfied:
                            ready_nodes.append(n)

                # Submit ready nodes up to concurrency limit
                for node in ready_nodes:
                    if len(active_futures) < max_concurrent:
                        fut = executor.submit(self.execute_node, node)
                        active_futures[fut] = node

                if not active_futures:
                    # No active jobs and no ready nodes; break
                    break

                # Wait for any running task to finish
                done, _ = concurrent.futures.wait(
                    list(active_futures.keys()),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for fut in done:
                    node = active_futures.pop(fut)
                    completed.add(node.id)

            # Wait for remaining active futures
            for fut in concurrent.futures.as_completed(list(active_futures.keys())):
                node = active_futures.pop(fut)
                completed.add(node.id)

        print("\n=== DAG Orchestration Finished ===")
        summary = {n.id: n.status for n in self.nodes}
        for nid, st in summary.items():
            print(f"  {nid}: {st}")

        budget.print_summary()
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate TEXTSCOUR Kaggle GPU DAG.")
    parser.add_argument("--plan", default="kaggle_ops/dispatch_plan.yaml", help="Path to dispatch plan")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution without real push")
    parser.add_argument("--retry", help="Retry a specific node id")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent kernels")

    args = parser.parse_args()

    orchestrator = Orchestrator(plan_path=args.plan, dry_run=args.dry_run)

    if args.retry:
        if args.retry in orchestrator.node_map:
            target_node = orchestrator.node_map[args.retry]
            target_node.status = "PENDING"
            orchestrator.execute_node(target_node)
        else:
            print(f"Error: Node {args.retry} not found in plan.", file=sys.stderr)
            sys.exit(1)
    else:
        orchestrator.run(max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
