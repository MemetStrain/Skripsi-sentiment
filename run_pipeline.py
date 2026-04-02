#!/usr/bin/env python3
"""
CPO Price Prediction Pipeline Orchestrator
Runs the full end-to-end pipeline from news scraping to price prediction.

Usage:
    python run_pipeline.py                          # Full pipeline, CSA model, auto-skip on
    python run_pipeline.py --model csa              # Explicit CSA model
    python run_pipeline.py --model improved         # Improved model
    python run_pipeline.py --model baseline         # Baseline model
    python run_pipeline.py --no-auto-skip           # Force re-run all steps
    python run_pipeline.py --start-from hmm         # Resume from HMM step
    python run_pipeline.py --only predict           # Run only prediction step
    python run_pipeline.py --force sentiment        # Force re-run sentiment even if output exists
    python run_pipeline.py --dry-run                # Show what would run without executing
    python run_pipeline.py --list-steps             # List all steps and exit
    python run_pipeline.py --quiet                  # Suppress subprocess stdout
"""

from __future__ import annotations

import argparse
import enum
import importlib.util
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Project root (directory containing this script)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# Enums / Data classes
# ---------------------------------------------------------------------------

class StepStatus(enum.Enum):
    PENDING = "PENDING"
    SKIPPED = "SKIPPED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED  = "FAILED"


@dataclass
class StepResult:
    status: StepStatus
    duration_seconds: float = 0.0
    skipped_reason: str = ""
    returncode: int = 0
    stderr_tail: str = ""
    error_message: str = ""


@dataclass
class PipelineStep:
    id: str
    name: str
    script_path: Path
    cwd: Path
    output_files: List[Path]
    description: str
    timeout_seconds: int = 3600


@dataclass
class PipelineConfig:
    model: str = "horizon"
    frequency: str = "daily"
    auto_skip: bool = True
    force_steps: List[str] = field(default_factory=list)
    explicit_skip_steps: List[str] = field(default_factory=list)
    verbose: bool = True
    dry_run: bool = False
    project_root: Path = field(default_factory=lambda: ROOT)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class PipelineLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._t0 = time.monotonic()
        # Create/truncate log file
        log_path.write_text("", encoding="utf-8")

    def _elapsed(self) -> str:
        secs = int(time.monotonic() - self._t0)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def log(self, msg: str = "", level: str = "INFO") -> None:
        line = f"[{self._elapsed()}] [{level:<7}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def step_banner(self, n: int, total: int, step_name: str) -> None:
        bar = "=" * 62
        self.log()
        self.log(bar)
        self.log(f"  Step {n}/{total}: {step_name}")
        self.log(bar)


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

def build_step_registry(config: PipelineConfig) -> List[PipelineStep]:
    root       = config.project_root
    news_dir   = root / "news"
    markov_dir = root / "markov"
    pred_dir   = root / "prediction"

    freq = config.frequency  # daily / weekly / monthly
    freq_cap = freq.capitalize()  # Daily / Weekly / Monthly

    model_scripts = {
        "horizon":  pred_dir / "horizon_forecast.py",
        "adaptive": pred_dir / "adaptive_prediction.py",
    }
    model_outputs = {
        "horizon":  pred_dir / "output_horizons" / freq_cap / "horizon_summary_{}.csv".format(freq_cap),
        "adaptive": pred_dir / "output" / "adaptive_prediction_results_{}.csv".format(freq_cap),
    }

    return [
        PipelineStep(
            id="scrape",
            name="News Scraping (MPOB)",
            script_path=news_dir / "scrap_fast.py",
            cwd=news_dir,
            output_files=[news_dir / "mpob_news_fast.csv"],
            description="Scrapes MPOB news articles from the web (auto-resumes from last date)",
            timeout_seconds=7200,  # 2 hours
        ),
        PipelineStep(
            id="preprocess",
            name="News Preprocessing",
            script_path=news_dir / "news_preprocessing.py",
            cwd=news_dir,
            output_files=[news_dir / "mpob_news_preprocessed.csv"],
            description="Cleans and normalizes raw news text (strips HTML, URLs, special chars)",
            timeout_seconds=600,
        ),
        PipelineStep(
            id="sentiment",
            name="FinBERT Sentiment Analysis",
            script_path=news_dir / "finbert_sentiment_analysis_flexible.py",
            cwd=news_dir,
            output_files=[
                news_dir / "mpob_news_with_sentiment.csv",
                news_dir / "output" / f"sentiment_aggregate_{freq_cap}.csv",
            ],
            description="Runs FinBERT transformer for sentiment scoring (GPU recommended; may take hours)",
            timeout_seconds=28800,  # 8 hours
        ),
        PipelineStep(
            id="hmm",
            name="HMM Market State Analysis",
            script_path=markov_dir / "cpo_hmm_states.py",
            cwd=markov_dir,
            output_files=[
                markov_dir / "output" / f"hmm_states_results_{freq}.csv",
            ],
            description="Trains Hidden Markov Model to identify market regimes (Bullish/Bearish/Neutral)",
            timeout_seconds=1800,
        ),
        PipelineStep(
            id="dataset",
            name="Feature Dataset Creation",
            script_path=root / "create_prediction_dataset.py",
            cwd=root,
            output_files=[
                markov_dir / f"cpo_prediction_dataset_{freq}.csv",
            ],
            description="Combines CPO prices, HMM states, sentiment, and technical indicators (~60 features)",
            timeout_seconds=1800,
        ),
        PipelineStep(
            id="predict",
            name=f"Price Prediction ({config.model.upper()})",
            script_path=model_scripts.get(config.model, model_scripts["horizon"]),
            cwd=pred_dir,
            output_files=[model_outputs.get(config.model, model_outputs["horizon"])],
            description=f"Trains and evaluates {config.model} models with horizon forecasting",
            timeout_seconds=14400,
        ),
    ]


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES = [
    ("pandas",       "pandas"),
    ("numpy",        "numpy"),
    ("sklearn",      "scikit-learn"),
    ("xgboost",      "xgboost"),
    ("transformers", "transformers"),
    ("torch",        "torch"),
    ("hmmlearn",     "hmmlearn"),
    ("bs4",          "beautifulsoup4"),
    ("requests",     "requests"),
    ("tqdm",         "tqdm"),
    ("matplotlib",   "matplotlib"),
    ("seaborn",      "seaborn"),
]

REQUIRED_DATA_FILES = [
    ROOT / "cpo" / "Data_CPO_Daily.csv",
    ROOT / "cpo" / "Data_CPO_Monthly.csv",
]


def check_prerequisites(config: PipelineConfig) -> List[str]:
    problems: List[str] = []

    # Python version
    if sys.version_info < (3, 8):
        problems.append(f"Python 3.8+ required, found {sys.version.split()[0]}")

    # Required packages (use find_spec to avoid triggering module-level code)
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                problems.append(f"Missing package: {pip_name}  ->  pip install {pip_name}")
        except (ModuleNotFoundError, ValueError):
            problems.append(f"Missing package: {pip_name}  ->  pip install {pip_name}")

    # Required base data files
    for path in REQUIRED_DATA_FILES:
        if not path.exists():
            problems.append(f"Missing data file: {path}")

    return problems


def ensure_output_dirs(config: PipelineConfig) -> None:
    dirs = [
        config.project_root / "news" / "output",
        config.project_root / "markov" / "output",
        config.project_root / "prediction" / "output",
        config.project_root / "prediction" / "output" / "csa_results",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------

def should_skip_step(
    step: PipelineStep, config: PipelineConfig
) -> Tuple[bool, str]:
    # Explicit --force overrides everything
    if step.id in config.force_steps:
        return False, "forced by --force flag"

    # Explicit --skip-* flag
    if step.id in config.explicit_skip_steps:
        return True, "explicitly skipped by --skip flag"

    # Auto-skip disabled
    if not config.auto_skip:
        return False, "auto-skip disabled"

    # Check all output files exist and are non-empty
    missing = [
        f for f in step.output_files
        if not f.exists() or f.stat().st_size == 0
    ]
    if missing:
        names = [f.name for f in missing]
        return False, f"missing output: {names}"

    names = [f.name for f in step.output_files]
    return True, f"output files already exist: {names}"


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def _last_n_lines(text: str, n: int) -> str:
    lines = (text or "").strip().splitlines()
    return "\n".join(lines[-n:]) if lines else ""


def _format_duration(seconds: float) -> str:
    secs = int(seconds)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def run_step(
    step: PipelineStep, config: PipelineConfig, logger: PipelineLogger
) -> StepResult:
    if config.dry_run:
        logger.log(f"[DRY RUN] Would run: {step.script_path}")
        logger.log(f"[DRY RUN] cwd       : {step.cwd}")
        return StepResult(status=StepStatus.SKIPPED, skipped_reason="dry-run mode")

    t_start = time.monotonic()

    try:
        proc = subprocess.run(
            [sys.executable, str(step.script_path)],
            cwd=str(step.cwd),
            timeout=step.timeout_seconds,
            # In verbose mode pass through stdout/stderr directly so user sees
            # real-time output (tqdm bars, training logs, etc.).
            # In quiet mode capture for error reporting only.
            stdout=None if config.verbose else subprocess.PIPE,
            stderr=None if config.verbose else subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",  # Windows cp1252 / tqdm safety
        )
        duration = time.monotonic() - t_start

        if proc.returncode != 0:
            tail = _last_n_lines(proc.stderr or "", 30)
            return StepResult(
                status=StepStatus.FAILED,
                duration_seconds=duration,
                returncode=proc.returncode,
                stderr_tail=tail,
                error_message=f"Process exited with code {proc.returncode}",
            )

        return StepResult(
            status=StepStatus.SUCCESS,
            duration_seconds=duration,
            returncode=0,
        )

    except subprocess.TimeoutExpired:
        duration = time.monotonic() - t_start
        hours = step.timeout_seconds / 3600
        return StepResult(
            status=StepStatus.FAILED,
            duration_seconds=duration,
            error_message=(
                f"Step timed out after {step.timeout_seconds}s ({hours:.1f}h). "
                f"Increase timeout_seconds in build_step_registry() if needed."
            ),
        )

    except FileNotFoundError:
        return StepResult(
            status=StepStatus.FAILED,
            duration_seconds=0.0,
            error_message=f"Script not found: {step.script_path}",
        )

    except Exception as exc:  # noqa: BLE001
        duration = time.monotonic() - t_start
        return StepResult(
            status=StepStatus.FAILED,
            duration_seconds=duration,
            error_message=f"Unexpected error: {exc}",
        )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    steps: List[PipelineStep],
    config: PipelineConfig,
    logger: PipelineLogger,
) -> Dict[str, StepResult]:
    results: Dict[str, StepResult] = {}
    total = len(steps)

    for i, step in enumerate(steps, 1):
        logger.step_banner(i, total, step.name)
        logger.log(f"Script : {step.script_path}")
        logger.log(f"CWD    : {step.cwd}")
        logger.log(f"Info   : {step.description}")

        skip, reason = should_skip_step(step, config)

        if skip:
            logger.log(f"SKIPPING — {reason}", "SKIP")
            results[step.id] = StepResult(
                status=StepStatus.SKIPPED,
                skipped_reason=reason,
            )
            continue

        logger.log("Starting step ...", "RUN")
        result = run_step(step, config, logger)
        results[step.id] = result

        dur = _format_duration(result.duration_seconds)
        if result.status == StepStatus.SUCCESS:
            logger.log(f"DONE — {dur}", "SUCCESS")
        elif result.status == StepStatus.SKIPPED:
            logger.log(f"SKIPPED — {result.skipped_reason}", "SKIP")
        else:
            logger.log(f"FAILED — {result.error_message}", "FAILED")
            if result.stderr_tail:
                logger.log("Last output from subprocess:", "FAILED")
                for line in result.stderr_tail.splitlines():
                    logger.log(f"  {line}", "FAILED")
            logger.log(
                f"To resume from this step after fixing the issue:",
                "TIP",
            )
            logger.log(
                f"  python run_pipeline.py --start-from {step.id} --model {config.model}",
                "TIP",
            )
            break  # Stop pipeline on first failure

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    steps: List[PipelineStep],
    results: Dict[str, StepResult],
    total_elapsed: float,
    logger: PipelineLogger,
    config: PipelineConfig,
) -> None:
    bar = "=" * 62
    logger.log()
    logger.log(bar)
    logger.log("  PIPELINE SUMMARY")
    logger.log(bar)

    col_step = 26
    col_status = 10
    col_dur = 14
    header = (
        f"  {'Step':<{col_step}} {'Status':<{col_status}} {'Duration':<{col_dur}} Notes"
    )
    logger.log(header)
    logger.log("  " + "-" * 58)

    for step in steps:
        r = results.get(step.id)
        if r is None:
            status_str = "NOT RUN"
            dur_str = "--"
            notes = ""
        else:
            status_str = r.status.value
            if r.status in (StepStatus.SKIPPED, StepStatus.PENDING):
                dur_str = "--"
            else:
                dur_str = _format_duration(r.duration_seconds)
            if r.status == StepStatus.SKIPPED:
                notes = r.skipped_reason[:40]
            elif r.status == StepStatus.FAILED:
                notes = r.error_message[:40]
            else:
                notes = ""

        name_trunc = step.name[:col_step]
        row = (
            f"  {name_trunc:<{col_step}} "
            f"{status_str:<{col_status}} "
            f"{dur_str:<{col_dur}} "
            f"{notes}"
        )
        logger.log(row)

    logger.log()
    logger.log(f"  Total elapsed : {_format_duration(total_elapsed)}")
    logger.log(f"  Model used    : {config.model}")
    logger.log(f"  Frequency     : {config.frequency}")

    # Show key output files
    any_success = any(
        r.status == StepStatus.SUCCESS for r in results.values()
    )
    if any_success:
        logger.log()
        logger.log("  Key output files:")
        freq = config.frequency
        freq_cap = freq.capitalize()
        root = config.project_root
        output_candidates = [
            root / "news" / "output" / f"sentiment_aggregate_{freq_cap}.csv",
            root / "markov" / "output" / f"hmm_states_results_{freq}.csv",
            root / "markov" / f"cpo_prediction_dataset_{freq}.csv",
            root / "prediction" / "output" / "prediction_results.csv",
            root / "prediction" / "output" / "prediction_results_improved.csv",
            root / "prediction" / "output" / "prediction_results_csa.csv",
        ]
        for p in output_candidates:
            if p.exists() and p.stat().st_size > 0:
                rel = p.relative_to(root)
                size_mb = p.stat().st_size / (1024 * 1024)
                logger.log(f"    {rel}  ({size_mb:.1f} MB)")

    logger.log(bar)

    failed = any(
        r.status == StepStatus.FAILED for r in results.values()
    )
    if failed:
        logger.log("  STATUS: FAILED — see errors above", "FAILED")
    else:
        logger.log("  STATUS: COMPLETED SUCCESSFULLY")
    logger.log(bar)
    logger.log(f"  Full log saved to: {logger.log_path}")
    logger.log(bar)


# ---------------------------------------------------------------------------
# Arg parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="End-to-end CPO price prediction pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Pipeline steps (in order):
          scrape      → News scraping from MPOB website
          preprocess  → News text cleaning
          sentiment   → FinBERT sentiment analysis (GPU recommended)
          hmm         → Hidden Markov Model market state identification
          dataset     → Feature dataset creation (~60 features)
          predict     → Price prediction model training + evaluation

        Examples:
          python run_pipeline.py                           # Full pipeline, CSA model
          python run_pipeline.py --model improved          # Use improved model
          python run_pipeline.py --no-auto-skip            # Force re-run all steps
          python run_pipeline.py --start-from hmm          # Resume from HMM step
          python run_pipeline.py --only predict --model csa
          python run_pipeline.py --force sentiment         # Re-run sentiment only
          python run_pipeline.py --dry-run                 # Preview without running
          python run_pipeline.py --list-steps              # Show step info and exit
        """),
    )

    grp_model = p.add_argument_group("Model selection")
    grp_model.add_argument(
        "--model",
        choices=["baseline", "improved", "csa"],
        default="csa",
        help="Prediction model to use (default: csa)",
    )
    grp_model.add_argument(
        "--frequency",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="Data frequency for HMM and dataset steps (default: daily)",
    )

    grp_skip = p.add_argument_group("Step control")
    grp_skip.add_argument(
        "--auto-skip",
        dest="auto_skip",
        action="store_true",
        default=True,
        help="Skip steps whose output files already exist (default: on)",
    )
    grp_skip.add_argument(
        "--no-auto-skip",
        dest="auto_skip",
        action="store_false",
        help="Force re-run all steps regardless of existing outputs",
    )
    grp_skip.add_argument(
        "--start-from",
        choices=["scrape", "preprocess", "sentiment", "hmm", "dataset", "predict"],
        metavar="STEP",
        help="Skip all steps before STEP (use to resume a failed run)",
    )
    grp_skip.add_argument(
        "--only",
        choices=["scrape", "preprocess", "sentiment", "hmm", "dataset", "predict"],
        metavar="STEP",
        help="Run only this single step",
    )
    grp_skip.add_argument(
        "--force",
        nargs="+",
        metavar="STEP",
        help="Force re-run specific step(s) even if output exists (e.g. --force sentiment hmm)",
    )
    grp_skip.add_argument("--skip-scrape",     action="store_true", help="Skip the scraping step")
    grp_skip.add_argument("--skip-preprocess", action="store_true", help="Skip the preprocessing step")
    grp_skip.add_argument("--skip-sentiment",  action="store_true", help="Skip the sentiment analysis step")
    grp_skip.add_argument("--skip-hmm",        action="store_true", help="Skip the HMM step")
    grp_skip.add_argument("--skip-dataset",    action="store_true", help="Skip the dataset creation step")

    grp_run = p.add_argument_group("Run options")
    grp_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without actually running anything",
    )
    grp_run.add_argument(
        "--quiet",
        action="store_true",
        help="Capture subprocess output (only shown on failure); default streams it live",
    )
    grp_run.add_argument(
        "--no-prereq-check",
        action="store_true",
        help="Skip Python package and data file prerequisite checks",
    )
    grp_run.add_argument(
        "--list-steps",
        action="store_true",
        help="Print all pipeline steps with descriptions and exit",
    )

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Handle --list-steps early (no logging needed)
    if args.list_steps:
        config = PipelineConfig()
        steps = build_step_registry(config)
        print("\nCPO Pipeline Steps\n" + "=" * 50)
        for i, s in enumerate(steps, 1):
            print(f"\n  {i}. [{s.id}]  {s.name}")
            print(f"     Script : {s.script_path}")
            print(f"     Info   : {s.description}")
            print(f"     Output : {[f.name for f in s.output_files]}")
            timeout_h = s.timeout_seconds / 3600
            print(f"     Timeout: {timeout_h:.1f}h")
        print()
        sys.exit(0)

    # Build config
    explicit_skip: List[str] = []
    if args.skip_scrape:     explicit_skip.append("scrape")
    if args.skip_preprocess: explicit_skip.append("preprocess")
    if args.skip_sentiment:  explicit_skip.append("sentiment")
    if args.skip_hmm:        explicit_skip.append("hmm")
    if args.skip_dataset:    explicit_skip.append("dataset")

    config = PipelineConfig(
        model=args.model,
        frequency=args.frequency,
        auto_skip=args.auto_skip,
        force_steps=args.force or [],
        explicit_skip_steps=explicit_skip,
        verbose=not args.quiet,
        dry_run=args.dry_run,
    )

    # Build full step list
    all_steps = build_step_registry(config)

    # Apply --start-from
    steps = all_steps
    if args.start_from:
        ids = [s.id for s in all_steps]
        try:
            idx = ids.index(args.start_from)
            steps = all_steps[idx:]
        except ValueError:
            print(f"ERROR: Unknown step id '{args.start_from}'", file=sys.stderr)
            sys.exit(1)

    # Apply --only
    if args.only:
        steps = [s for s in all_steps if s.id == args.only]
        if not steps:
            print(f"ERROR: Unknown step id '{args.only}'", file=sys.stderr)
            sys.exit(1)

    # Set up logger
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / f"pipeline_run_{ts}.log"
    logger = PipelineLogger(log_path)

    # Header
    logger.log("=" * 62)
    logger.log("  CPO Price Prediction Pipeline Orchestrator")
    logger.log("=" * 62)
    logger.log(f"Started  : {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.log(f"Model    : {config.model}")
    logger.log(f"Frequency: {config.frequency}")
    logger.log(f"Steps    : {[s.id for s in steps]}")
    logger.log(f"Auto-skip: {config.auto_skip}")
    logger.log(f"Dry-run  : {config.dry_run}")
    logger.log(f"Log file : {log_path}")

    # Prerequisite check
    if not args.no_prereq_check:
        logger.log()
        logger.log("Checking prerequisites ...")
        problems = check_prerequisites(config)
        hard_errors = [p for p in problems if "Missing package" in p]
        soft_warns  = [p for p in problems if "Missing package" not in p]
        for w in soft_warns:
            logger.log(f"  WARN: {w}", "WARN")
        for e in hard_errors:
            logger.log(f"  ERROR: {e}", "FAILED")
        if hard_errors:
            logger.log("Prerequisite check failed — install missing packages and retry.", "FAILED")
            sys.exit(1)
        logger.log("Prerequisites OK.")
        ensure_output_dirs(config)

    # Run pipeline
    t0 = time.monotonic()
    results = run_pipeline(steps, config, logger)
    total_elapsed = time.monotonic() - t0

    # Summary
    print_summary(all_steps, results, total_elapsed, logger, config)

    # Exit code
    failed = any(r.status == StepStatus.FAILED for r in results.values())
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
