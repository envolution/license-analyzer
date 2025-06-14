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
    TimeRemainingColumn,
)
from rich.status import Status
from rich.live import Live
from rich.padding import Padding
from rich.logging import RichHandler
from rich.filesize import decimal


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
    level=logging.INFO,  # Default to INFO, can be WARNING if not verbose
    format="%(message)s",
    handlers=[
        RichHandler(console=console, show_time=True, show_level=True, show_path=False)
    ],
)
logger = logging.getLogger(__name__)  # Use standard logger, RichHandler formats it


def format_text_output(file_path: str, matches: List[LicenseMatch]) -> str:
    """Format matches as human-readable text."""
    # file_path is the path of the analyzed file (e.g., "LICENSE.txt")
    # match.name is the SPDX license ID (e.g., "MIT", "Apache-2.0"), which no longer contains .txt
    output = [f"[bold green]Analysis results for: {file_path}[/bold green]"]
    output.append("-" * 60)

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
                f"method: [magenta]{match.method.value}[/magenta]"
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
            }
            for match in matches
        ]

    return json.dumps(json_results, indent=2)


def format_csv_output(results: dict) -> str:
    """Format results as CSV."""
    lines = [
        "file_path,license_name,score,method"
    ]  # Removed 'license_type' as it's not present in Match

    for file_path, matches in results.items():
        for match in matches:
            lines.append(
                f'"{file_path}","{match.name}",{match.score},{match.method.value}'
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
        updater = LicenseUpdater(
            cache_dir=DEFAULT_CACHE_BASE_DIR,
            spdx_data_dir=args.spdx_dir,
        )

        update_performed = False
        update_message = ""

        # --- Handle update check with Rich Progress ---
        # Using a Live context for a transient status message when total is None
        # and a Progress bar when total is known (downloading/extracting)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",  # Percentage
            "•",
            TimeElapsedColumn(),
            "•",
            TransferSpeedColumn(),  # Download speed
            "•",
            TimeRemainingColumn(),  # Time remaining for downloads
            console=console,
            transient=True,  # Hide progress bar when done
        ) as progress:
            updater_task_id = progress.add_task(
                "[cyan]Checking for license updates...",
                total=None,  # Start as indeterminate
            )

            def updater_progress_callback(current, total, status_msg):
                # If total is 0 (or None), it's indeterminate progress (e.g., "Checking for updates...")
                # If total > 0, it's determinate progress (downloading, extracting files)
                progress.update(
                    updater_task_id,
                    total=total
                    if total > 0
                    else None,  # Set total to None for indeterminate spinner
                    completed=current,
                    description=f"[cyan]{status_msg}",  # status_msg is plain text from updater
                )

            # Perform the update check
            # This call will use the updater_progress_callback for its updates
            update_performed, update_message = updater.check_for_updates(
                force=args.update, progress_callback=updater_progress_callback
            )

            # Ensure the task is completed or removed, especially if it was indeterminate
            if progress.tasks[updater_task_id].total is None:
                progress.update(
                    updater_task_id,
                    total=1,
                    completed=1,
                    description=f"[cyan]{update_message}",
                )
            progress.remove_task(updater_task_id)  # Explicitly remove for clean exit

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

        # --- Initialize Analyzer and build database with Rich Progress ---
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
                # status_msg is now plain text from core.py
                db_progress.update(
                    db_task,
                    total=total,
                    completed=current,
                    description=f"[green]Building license cache: {status_msg}",
                )

            analyzer = LicenseAnalyzer(
                spdx_dir=args.spdx_dir,
                cache_dir=args.cache_dir,
                embedding_model_name=args.embedding_model,
                db_progress_callback=db_progress_callback,
            )
        console.print("[bold green]✔ License database is ready.[/bold green]")

        # --- Analyze files with Rich Progress (for multiple files) or Status (for single file) ---
        if args.files:
            file_paths_to_analyze = [Path(f) for f in args.files]

            if len(file_paths_to_analyze) == 1:
                # For a single file, a simple Status spinner is more elegant
                file_path = file_paths_to_analyze[0]

                # Using Live for Status to ensure it cleans up correctly even on errors
                with Status(
                    f"[cyan]Analyzing [bold]{file_path.name}[/bold]...",
                    spinner="dots",
                    console=console,
                ) as status_message:
                    # Define a dummy per_entry_embed_callback to update the Status message
                    def per_entry_embed_callback(status_msg: str):
                        status_message.update(
                            f"[cyan]Analyzing [bold]{file_path.name}[/bold]: {status_msg}"
                        )

                    matches = analyzer.analyze_file(
                        file_path,
                        args.top_n,
                        per_entry_embed_callback=per_entry_embed_callback,
                    )
                    matches = [m for m in matches if m.score >= args.min_score]
                    results = {str(file_path): matches}
                console.print("\n")  # Newline after single file analysis status
            else:
                # For multiple files, use a Progress bar
                with Progress(
                    SpinnerColumn(),
                    TextColumn(
                        "[progress.description]{task.description}"
                    ),  # This will show the dynamic messages
                    BarColumn(),
                    "{task.completed} of {task.total}",
                    console=console,
                    transient=True,
                ) as analysis_progress:
                    analysis_task = analysis_progress.add_task(
                        "[cyan]Analyzing license files...",
                        total=len(file_paths_to_analyze),
                    )

                    def analysis_progress_callback(current, total, status_msg):
                        # status_msg comes from LicenseAnalyzer and can include embedding details
                        analysis_progress.update(
                            analysis_task,
                            completed=current,
                            description=f"[cyan]{status_msg}",
                        )

                    results = analyzer.analyze_multiple_files(
                        file_paths_to_analyze,
                        args.top_n,
                        analysis_progress_callback=analysis_progress_callback,
                    )
                    # Filter by minimum score for all files after analysis
                    for file_path in results:
                        results[file_path] = [
                            m for m in results[file_path] if m.score >= args.min_score
                        ]
                console.print("[bold green]✔ Finished analyzing files.[/bold green]")

            # Analysis done, print results
            console.print("\n")  # Add a newline after progress bar for cleaner output
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
                    f"\n[bold magenta]Database stats:[/bold magenta] "
                    f"[blue]{stats['total_licenses']}[/blue] licenses loaded.",
                )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
