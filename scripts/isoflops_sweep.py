from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from cs336_scaling.client import (
    get_budget,
    get_experiment,
    list_experiments,
    submit_experiment,
)
from cs336_scaling.training.model.basic_model import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig, WarmupCosineDecay
from cs336_scaling.training.training_config import TrainingConfig


@dataclass(frozen=True)
class Architecture:
    name: str
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def n_nonembed(self) -> int:
        return 12 * self.num_hidden_layers * self.hidden_size**2


# Same shape family as the LR/BS sweep: head_dim=64, d/L=64, intermediate ≈ 8/3*d.
ARCHES: dict[str, Architecture] = {
    "XXS": Architecture("XXS", 256, 4, 4, 704),    # N_nonembed = 3.15M
    "XS":  Architecture("XS",  320, 5, 5, 896),    # N_nonembed = 6.14M
    "S":   Architecture("S",   384, 6, 6, 1024),   # N_nonembed = 10.6M
    "M":   Architecture("M",   512, 8, 8, 1344),   # N_nonembed = 25.2M
    "L":   Architecture("L",   640, 10, 10, 1728), # N_nonembed = 49.2M
    "XL":  Architecture("XL",  768, 12, 12, 2048), # N_nonembed = 84.9M
    "XXL": Architecture("XXL", 1024, 16, 16, 2752),# N_nonembed = 201M
    # "Final": Architecture("Final", 1536, 24, 24, 4096),# N_nonembed = 679.5M
    "Final": Architecture("Final", 1472, 23, 23, 3904),# N_nonembed = 598.0M
    # "Final": Architecture("Final", 1408, 22, 22, 3776),# N_nonembed = 523.4M
    # "Final": Architecture("Final", 1344, 21, 21, 3392),# N_nonembed = 455.2M
    # "Final": Architecture("Final", 1280, 20, 20, 3584),# N_nonembed = 679.5M
}

SEQ_LEN = 512
N_EVALS = 16

def token_unit(train_batch_size: int) -> int:   # basic unit of D increment
    return SEQ_LEN * train_batch_size * N_EVALS


def _round_d(target_d: float, train_batch_size: int) -> int:   # round D to the nearest multiple of token_unit
    unit = token_unit(train_batch_size)
    return max(unit, round(target_d / unit) * unit)


def make_point(
    arch_name: str, target_C: float, train_batch_size: int
) -> IsoflopsPoint:  # Pick D such that 6 * N * D = target_C and rounded to satisfy divisibility
    arch = ARCHES[arch_name]
    target_d = target_C / (6.0 * arch.n_nonembed)
    return IsoflopsPoint(arch_name, _round_d(target_d, train_batch_size))


@dataclass(frozen=True)
class IsoflopsPoint:
    arch_name: str
    total_train_tokens: int

    def n_nonembed(self) -> int:
        return ARCHES[self.arch_name].n_nonembed

    def actual_C(self) -> float:
        return 6.0 * self.n_nonembed() * self.total_train_tokens


@dataclass(frozen=True)
class IsoflopsScale:
    name: str
    target_C: float
    points: tuple[IsoflopsPoint, ...]
    peak_lr: float
    train_batch_size: int
    default_max_runtime_seconds: float


# Initial scales. Start with 3 points per scale; add new point until the raw points are in U-shape.
SCALES: dict[str, IsoflopsScale] = {
    "XS": IsoflopsScale(
        name="XS",
        target_C=1e16,
        points=(
            make_point("M",   1e16, 64),
            make_point("S",   1e16, 64),
            make_point("XS",  1e16, 64),
            make_point("XXS", 1e16, 64),
        ),
        peak_lr=1e-2,
        train_batch_size=64,
        default_max_runtime_seconds=1000.0,
    ),
    "S": IsoflopsScale(
        name="S",
        target_C=3e16,
        points=(
            # make_point("L", 3e16, 64),
            # make_point("M", 3e16, 64),
            # make_point("S", 3e16, 64),
            make_point("XS",  3e16, 64),
        ),
        peak_lr=1e-2,
        train_batch_size=64,
        default_max_runtime_seconds=1000.0,
    ),
    "M": IsoflopsScale(
        name="M",
        target_C=1e17,
        points=(
            make_point("XL", 1e17, 64),
            make_point("L",  1e17, 64),
            make_point("M",  1e17, 64),
            make_point("S",  1e17, 64),
        ),
        peak_lr=1e-2,
        train_batch_size=64,
        default_max_runtime_seconds=4000.0,
    ),
    "L": IsoflopsScale(
        name="L",
        target_C=3e17,
        points=(
            # make_point("XXL", 3e17, 128),
            make_point("XL",  3e17, 128),
            make_point("L",   3e17, 128),
            make_point("M",   3e17, 128),
        ),
        peak_lr=8e-3,  # nudge down for safety
        train_batch_size=128,
        default_max_runtime_seconds=5000,
    ),
    "XL": IsoflopsScale(
        name="XL",
        target_C=1e18,
        points=(
            make_point("L",   1e18, 128),
            make_point("XL",  1e18, 128),
            make_point("XXL", 1e18, 128),
        ),
        peak_lr=6e-3,   # nudged down further
        train_batch_size=128,
        default_max_runtime_seconds=7200.0,
    ),
    "Final": IsoflopsScale(
        name="Final",
        target_C=7e19,
        points=(
            make_point("Final", 7e19, 64),
        ),
        peak_lr=5e-3,
        train_batch_size=64,
        default_max_runtime_seconds=1800.0,
    ),
}


def build_config(
    arch: Architecture,
    total_train_tokens: int,
    peak_lr: float,
    train_batch_size: int,
    max_runtime_seconds: float,
    model_seed: int = 0,
) -> TrainingConfig:
    return TrainingConfig(
        architecture_config=BasicTransformerConfig(
            attention_bias=False,
            head_dim=arch.head_dim,
            hidden_size=arch.hidden_size,
            intermediate_size=arch.intermediate_size,
            num_attention_heads=arch.num_attention_heads,
            num_hidden_layers=arch.num_hidden_layers,
            num_key_value_heads=arch.num_attention_heads,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000,
            tie_word_embeddings=False,
            dtype="bfloat16",
            vocab_size=32_000,
        ),
        optimizer_config=AdamWConfig(
            lr_scheduler=WarmupCosineDecay(
                peak_value=peak_lr,
                final_lr_frac=0.1,
                warmup_frac=0.05,
                init_value=0.0,
            ),
            weight_decay=1e-2,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            eps_root=1e-8,
            grad_clip_norm=1.0,
        ),
        train_batch_size=train_batch_size,
        val_batch_size=64,
        n_evals=16,
        total_train_tokens=total_train_tokens,
        max_runtime_seconds=max_runtime_seconds,
        model_seed=model_seed,
    )


def load_results(results_path: Path, scale: IsoflopsScale) -> dict:
    if results_path.exists():
        with results_path.open() as f:
            return json.load(f)
    return {
        "scale": scale.name,
        "target_C": scale.target_C,
        "peak_lr": scale.peak_lr,
        "train_batch_size": scale.train_batch_size,
        "runs": [],
    }


def save_results(results_path: Path, results: dict) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)


def index_existing_experiments() -> dict[str, int]:
    out: dict[str, int] = {}
    for exp in list_experiments():
        try:
            out[exp.training_config.unique_id] = exp.experiment_id
        except Exception:
            continue
    return out


def submit_grid(
    scale: IsoflopsScale,
    points: list[IsoflopsPoint],
    max_runtime_seconds: float,
    results_path: Path,
) -> dict:
    results = load_results(results_path, scale)
    runs_by_key = {
        (r["arch_name"], r["total_train_tokens"]): r for r in results["runs"]
    }
    hash_to_id = index_existing_experiments()

    for point in points:
        arch = ARCHES[point.arch_name]
        key = (point.arch_name, point.total_train_tokens)
        
        existing_local = runs_by_key.get(key)
        if existing_local and existing_local.get("experiment_id") is not None:
            print(
                f"tracked: {point.arch_name:>4s} D={point.total_train_tokens:>11d} "
                f"id={existing_local['experiment_id']}"
            )
            continue

        config = build_config(
            arch,
            point.total_train_tokens,
            scale.peak_lr,
            scale.train_batch_size,
            max_runtime_seconds,
        )
        existing_id = hash_to_id.get(config.unique_id)
        if existing_id is not None:
            print(
                f"found in API: {point.arch_name:>4s} "
                f"D={point.total_train_tokens:>11d} id={existing_id}"
            )
            run = _new_run_record(point, scale, existing_id, "queued")
        else:
            try:
                resp = submit_experiment(config)
            except RuntimeError as e:
                print(f"ERROR {point.arch_name} D={point.total_train_tokens}: {e}")
                continue
            print(
                f"submitted: {point.arch_name:>4s} "
                f"D={point.total_train_tokens:>11d} "
                f"id={resp.experiment_id} "
                f"(remaining {resp.budget_summary.remaining_seconds:.0f}s)"
            )
            run = _new_run_record(point, scale, resp.experiment_id, "queued")

        runs_by_key[key] = run
        results["runs"] = list(runs_by_key.values())
        save_results(results_path, results)

    return results


def _new_run_record(
    point: IsoflopsPoint,
    scale: IsoflopsScale,
    experiment_id: int,
    status: str,
) -> dict:
    arch = ARCHES[point.arch_name]
    return {
        "arch_name": point.arch_name,
        "hidden_size": arch.hidden_size,
        "num_hidden_layers": arch.num_hidden_layers,
        "num_attention_heads": arch.num_attention_heads,
        "head_dim": arch.head_dim,
        "intermediate_size": arch.intermediate_size,
        "n_nonembed": arch.n_nonembed,
        "total_train_tokens": point.total_train_tokens,
        "actual_C": point.actual_C(),
        "peak_lr": scale.peak_lr,
        "train_batch_size": scale.train_batch_size,
        "experiment_id": experiment_id,
        "status": status,
        "final_val_loss": None,
        "used_runtime_seconds": None,
        "val_losses": None,
    }


def poll_until_done(results_path: Path, poll_interval: int) -> dict:
    terminal = {"completed", "failed"}
    while True:
        with results_path.open() as f:
            results = json.load(f)
        pending = [r for r in results["runs"] if r["status"] not in terminal]
        if not pending:
            print("All runs reached a terminal status.")
            return results

        for r in pending:
            exp_id = r.get("experiment_id")
            if exp_id is None:
                continue
            exp = get_experiment(exp_id)
            r["status"] = exp.status.status_type
            if exp.status.status_type == "completed":
                r["final_val_loss"] = exp.status.val_losses[-1]
                r["used_runtime_seconds"] = exp.status.used_runtime_seconds
                r["val_losses"] = list(exp.status.val_losses)
            elif exp.status.status_type == "failed":
                r["used_runtime_seconds"] = exp.status.used_runtime_seconds
                reason = exp.status.reason
                if reason.reason == "timeout":
                    r["failure_reason"] = "timeout"
                    r["partial_val_losses"] = list(reason.partial_val_losses)
                else:
                    r["failure_reason"] = "unexpected"
                    r["failure_message"] = reason.failure
                r["val_losses"] = None

        save_results(results_path, results)

        n_done = sum(1 for r in results["runs"] if r["status"] in terminal)
        n_total = len(results["runs"])
        budget = get_budget()
        print(
            f"{n_done}/{n_total} done; "
            f"budget remaining {budget.remaining_seconds:.0f}s "
            f"({budget.remaining_seconds / 3600:.2f}h)"
        )
        if n_done == n_total:
            return results
        time.sleep(poll_interval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", required=True, choices=list(SCALES.keys()))
    ap.add_argument("--results-dir", default="results/isoflops")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll-only", action="store_true")
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=None,
        help="Per-run wall-clock cap (refunded on completion). "
        "Default: scale.default_max_runtime_seconds.",
    )
    args = ap.parse_args()

    scale = SCALES[args.scale]
    points = list(scale.points)
    max_runtime = (
        args.max_runtime_seconds
        if args.max_runtime_seconds is not None
        else scale.default_max_runtime_seconds
    )

    print(
        f"Isoflops scale {scale.name}: target_C={scale.target_C:.2e}, "
        f"LR={scale.peak_lr}, BS={scale.train_batch_size}"
    )
    print(
        f"  {'arch':>5s}  {'N_nonemb':>9s}  {'D':>11s}  {'D/N':>5s}  "
        f"{'actual_C':>10s}"
    )
    for p in points:
        arch = ARCHES[p.arch_name]
        d_over_n = p.total_train_tokens / arch.n_nonembed
        print(
            f"  {p.arch_name:>5s}  "
            f"{arch.n_nonembed:>9.2e}  "
            f"{p.total_train_tokens:>11d}  "
            f"{d_over_n:>5.1f}  "
            f"{p.actual_C():>10.2e}"
        )

    n_runs = len(points)
    total_reserved = n_runs * max_runtime
    print(
        f"\n{n_runs} runs; per-run max_runtime={max_runtime:.0f}s; "
        f"total reserved {total_reserved:.0f}s ({total_reserved / 60:.1f}min)"
    )

    budget = get_budget()
    print(
        f"Budget remaining: {budget.remaining_seconds:.0f}s "
        f"({budget.remaining_seconds / 3600:.2f}h)"
    )

    if not args.poll_only and total_reserved > budget.remaining_seconds:
        print(
            "ERROR: reservation exceeds remaining budget. "
            "Lower --max-runtime-seconds, drop points, or pick a smaller scale."
        )
        return

    if args.dry_run:
        print("\n[dry-run] not submitting.")
        return

    results_path = Path(args.results_dir) / f"scale_{scale.name}.json"
    print(f"\nResults file: {results_path}")

    if not args.poll_only:
        print("\nSubmitting points...")
        submit_grid(scale, points, max_runtime, results_path)

    print(f"\nPolling (every {args.poll_interval}s)...")
    results = poll_until_done(results_path, args.poll_interval)
    print(" Finished!!!")


if __name__ == "__main__":
    main()
