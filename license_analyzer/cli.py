# license_analyzer/cli.py
#!/usr/bin/env python3
"""
Command Line Interface for License Analyzer

Example usage:
    python -m license_analyzer.cli single_license.txt
    python -m license_analyzer.cli license1.txt license2.txt license3.txt
    license-analyzer --top-n 10 --format json license.txt
    license-analyzer --update --verbose # Force update and show details
    license-analyzer --spdx-dir /custom/spdx/path LICENSE
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List
import logging
import contextlib  # For redirect_stdout/stderr if needed for mocking

# Adjusted import to reflect package structure
from license_analyzer.core import LicenseAnalyzer, LicenseMatch
from license_analyzer.updater import LicenseUpdater
from appdirs import user_cache_dir

# NEW: Rich imports
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.status import Status
from rich.live import Live
from rich.padding import Padding
from rich.logging import RichHandler


# Default paths now managed by appdirs
APP_NAME = "license-analyzer"
APP_AUTHOR = "envolution"
DEFAULT_CACHE_BASE_DIR = Path(user_cache_dir(appname=APP_NAME, appauthor=APP_AUTHOR))
DEFAULT_SPDX_DATA_DIR = DEFAULT_CACHE_BASE_DIR / "spdx"
DEFAULT_DB_CACHE_DIR = DEFAULT_CACHE_BASE_DIR / "db_cache"

# Initialize Rich Console
console = Console()

# Configure logging to use RichHandler for better output in verbose mode
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(console=console, show_time=True, show_level=True, show_path=False)
    ],
)
logger = logging.getLogger(__name__)  # Use standard logger, RichHandler formats it


def format_text_output(file_path: str, matches: List[LicenseMatch]) -> str:
    """Format matches as human-readable text."""
    output = [f"[bold green]Analysis results for: {file_path}[/bold green]"]
    output.append("-" * 60)  # Changed to dashes for consistency with Rich styling

    if matches:
        for match in matches:
            # Use Rich markup for colored output
            score_color = (
                "red"
                if match.score < 0.7
                else ("yellow" if match.score < 0.9 else "green")
            )
            output.append(
                f"[cyan]{match.name:<30}[/cyan] score: [{score_color}]{match.score:.4f}[/{score_color}]  "
                f"method: [magenta]{match.method.value}[/magenta]  type: [blue]{match.license_type}[/blue]"
            )
    else:
        output.append("[italic yellow]No matches found.[/italic yellow]")

    return "\n".join(output)


def format_json_output(results: dict) -> str:
    """Format results as JSON."""
    json_results = {}

    for file_path, matches in results.items():
        json_results[file_path] = [
            {
                "name": match.name,
                "score": match.score,
                "method": match.method.value,
                "license_type": match.license_type,
            }
            for match in matches
        ]

    return json.dumps(json_results, indent=2)


def format_csv_output(results: dict) -> str:
    """Format results as CSV."""
    lines = ["file_path,license_name,score,method,license_type"]

    for file_path, matches in results.items():
        for match in matches:
            lines.append(
                f'"{file_path}","{match.name}",{match.score},'
                f"{match.method.value},{match.license_type}"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze license files to identify SPDX licenses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s LICENSE.txt
  %(prog)s --top-n 10 license1.txt license2.txt
  %(prog)s --format json --top-n 5 *.txt
  %(prog)s --update --verbose # Force update and show details
  %(prog)s --spdx-dir /custom/spdx/path LICENSE
        """,
    )

    parser.add_argument(
        "files",
        nargs="*",  # Changed to allow 0 files if only --update is used
        help="License files to analyze",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top matches to return per file (default: 5)",
    )

    parser.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--spdx-dir",
        type=Path,
        default=DEFAULT_SPDX_DATA_DIR,  # Updated default
        help=f"Path to SPDX licenses directory (default: {DEFAULT_SPDX_DATA_DIR})",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_DB_CACHE_DIR,  # Updated default for DB cache
        help=f"Path to cache directory for analyzer database (default: {DEFAULT_DB_CACHE_DIR})",
    )

    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name (default: all-MiniLM-L6-v2)",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score threshold for matches (default: 0.0)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--update",
        "-u",  # New argument for forced updates
        action="store_true",
        help="Force an update of the SPDX license database from GitHub",
    )

    args = parser.parse_args()

    # Set logging level for the RichHandler
    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.WARNING)

    try:
        # Initialize the updater first
        # It needs its own cache dir and the spdx data dir
        updater = LicenseUpdater(
            cache_dir=DEFAULT_CACHE_BASE_DIR,  # Base cache for updater's internal files
            spdx_data_dir=args.spdx_dir,  # Where actual SPDX licenses go
        )

        update_performed = False
        update_message = ""

        # --- Handle update check with Rich Progress ---
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "  {task.percentage:>3.0f}%",  # Adjusted percentage column
            TimeElapsedColumn(),
            TransferSpeedColumn(),  # For download speed
            console=console,
            transient=True,  # Hide progress bar when done
        ) as progress:
            # Define a callback for updater to report progress
            updater_task_id = progress.add_task(
                "[cyan]Checking for license updates...", total=None
            )

            def updater_progress_callback(current, total, status_msg):
                if total:  # If total is known (e.g., file size or count)
                    progress.update(
                        updater_task_id,
                        total=total,
                        completed=current,
                        description=f"[cyan]{status_msg}",
                    )
                else:  # For indeterminate progress (e.g., initial fetch)
                    # If total is 0 or 1, it's likely a phase change not a continuous download
                    if (
                        progress.tasks[updater_task_id].total is None
                        or progress.tasks[updater_task_id].total <= 1
                    ):
                        progress.update(
                            updater_task_id,
                            total=1,
                            completed=0,
                            description=f"[cyan]{status_msg}",
                        )
                    progress.update(
                        updater_task_id, advance=0.1
                    )  # Simulate progress for spinner
                    # Re-enable BarColumn and TaskProgressColumn if you want them for this phase
                    # For simple status, it's better to just update description and let spinner run.

            # Perform the update check
            update_performed, update_message = updater.check_for_updates(
                force=args.update, progress_callback=updater_progress_callback
            )
            progress.stop_task(updater_task_id)  # Stop the spinner/progress for updater

        if update_message:
            console.print(
                Padding(f"[bold]{update_message}[/bold]", (1, 0, 1, 0))
            )  # Add some padding

        if not args.files and update_performed:
            # If only update was performed and no files for analysis, exit
            sys.exit(0)
        elif not args.files and not update_performed:
            # If no files to analyze, and no update was performed explicitly or implicitly,
            console.print(
                "[yellow]No license files provided for analysis. Use --help for usage.[/yellow]"
            )
            sys.exit(0)

        # --- Initialize Analyzer and build database with Rich Status/Progress ---
        analyzer = None
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "{task.completed} of {task.total}",  # Show file count
            console=console,
            transient=True,
        ) as db_progress:
            db_task = db_progress.add_task(
                "[green]Building license cache...", total=None
            )

            def db_progress_callback(current, total, status_msg):
                # This callback is fired by LicenseDatabase._update_database
                db_progress.update(
                    db_task,
                    total=total,
                    completed=current,
                    description=f"[green]{status_msg}",
                )

            analyzer = LicenseAnalyzer(
                spdx_dir=args.spdx_dir,
                cache_dir=args.cache_dir,
                embedding_model_name=args.embedding_model,
                db_progress_callback=db_progress_callback,
            )
        console.print("[bold green]✔ License database is ready.[/bold green]")

        # --- Analyze files with Rich Status ---
        if args.files:
            with Status(
                "[cyan]Analyzing license file(s)...[/cyan]",
                spinner="moon",
                console=console,
            ) as analysis_status:
                if len(args.files) == 1:
                    file_path = Path(args.files[0])
                    analysis_status.update(
                        f"[cyan]Analyzing [bold]{file_path.name}[/bold]...[/cyan]"
                    )
                    matches = analyzer.analyze_file(file_path, args.top_n)

                    # Filter by minimum score
                    matches = [m for m in matches if m.score >= args.min_score]
                    results = {str(file_path): matches}
                else:
                    analysis_status.update(
                        f"[cyan]Analyzing {len(args.files)} license files...[/cyan]"
                    )
                    results = analyzer.analyze_multiple_files(args.files, args.top_n)

                    # Filter by minimum score
                    for file_path in results:
                        results[file_path] = [
                            m for m in results[file_path] if m.score >= args.min_score
                        ]
            # Analysis done, print results
            console.print("\n")  # Add a newline after spinner for cleaner output
            if args.format == "json":
                console.print(format_json_output(results))
            elif args.format == "csv":
                console.print(format_csv_output(results))
            else:  # text format
                for file_path, matches in results.items():
                    if len(results) > 1:
                        console.print(
                            Padding(
                                f"[bold grey]--- {file_path} ---[/bold grey]",
                                (1, 0, 0, 0),
                            )
                        )
                    console.print(format_text_output(file_path, matches))

            # Show database stats if verbose
            if args.verbose:
                stats = analyzer.get_database_stats()
                console.print(
                    f"\n[bold magenta]Database stats:[/bold magenta] [blue]{stats['licenses']}[/blue] licenses, "
                    f"[blue]{stats['exceptions']}[/blue] exceptions ([blue]{stats['total']}[/blue] total)",
                )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
