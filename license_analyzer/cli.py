# license_analyzer/cli.py
#!/usr/bin/env python3
"""
Command Line Interface for License Analyzer

Example usage:
    python -m license_analyzer.cli single_license.txt
    python -m license_analyzer.cli license1.txt license2.txt license3.txt
    license-analyzer --top-n 10 --format json license.txt
    license-analyzer --update # Force update license data
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List
import logging # Already present, but explicit

# Adjusted import to reflect package structure
from license_analyzer.core import LicenseAnalyzer, LicenseMatch
from license_analyzer.updater import LicenseUpdater # NEW import
from appdirs import user_cache_dir # NEW import

# Default paths now managed by appdirs
APP_NAME = "license-analyzer"
APP_AUTHOR = "envolution"
DEFAULT_CACHE_BASE_DIR = Path(user_cache_dir(appname=APP_NAME, appauthor=APP_AUTHOR))
DEFAULT_SPDX_DATA_DIR = DEFAULT_CACHE_BASE_DIR / "spdx"
DEFAULT_DB_CACHE_DIR = DEFAULT_CACHE_BASE_DIR / "db_cache"


def format_text_output(file_path: str, matches: List[LicenseMatch]) -> str:
    """Format matches as human-readable text."""
    output = [f"Analysis results for: {file_path}"]
    output.append("=" * 60)

    if matches:
        for match in matches:
            output.append(
                f"{match.name:<30} score: {match.score:.4f}  "
                f"method: {match.method.value}  type: {match.license_type}"
            )
    else:
        output.append("No matches found.")

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
                "license_type": match.license_type
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
                f'{match.method.value},{match.license_type}'
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
        """
    )

    parser.add_argument(
        "files",
        nargs="*", # Changed to allow 0 files if only --update is used
        help="License files to analyze"
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top matches to return per file (default: 5)"
    )

    parser.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text)"
    )

    parser.add_argument(
        "--spdx-dir",
        type=Path,
        default=DEFAULT_SPDX_DATA_DIR, # Updated default
        help=f"Path to SPDX licenses directory (default: {DEFAULT_SPDX_DATA_DIR})"
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_DB_CACHE_DIR, # Updated default for DB cache
        help=f"Path to cache directory for analyzer database (default: {DEFAULT_DB_CACHE_DIR})"
    )

    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name (default: all-MiniLM-L6-v2)"
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score threshold for matches (default: 0.0)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--update", "-u", # New argument for forced updates
        action="store_true",
        help="Force an update of the SPDX license database from GitHub"
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Initialize the updater first
        # It needs its own cache dir and the spdx data dir
        updater = LicenseUpdater(
            cache_dir=DEFAULT_CACHE_BASE_DIR, # Base cache for updater's internal files
            spdx_data_dir=args.spdx_dir # Where actual SPDX licenses go
        )

        update_performed = False
        if args.update:
            logging.info("Forcing SPDX license database update...")
            update_performed = updater.check_for_updates(force=True)
            if update_performed:
                print("SPDX license database updated successfully.")
            else:
                print("SPDX license database is already up-to-date or update failed. Check logs for details.")

            # If only update was requested and no files for analysis, exit
            if not args.files:
                sys.exit(0)
        else:
            # Perform daily update check if not forced and not only updating
            # This happens implicitly for interactive CLI runs
            if updater.check_for_updates(force=False):
                print("Note: SPDX license database was updated in the background.")
                update_performed = True


        # If there are no files to analyze, and no update was performed explicitly,
        # and no update happened implicitly, perhaps the user just ran `license-analyzer`
        # without args, and it's already up-to-date. In this case, we might want to exit
        # or give a hint.
        if not args.files and not update_performed:
            print("No license files provided for analysis. Use --help for usage.")
            sys.exit(0)
        elif not args.files and update_performed:
            # Only update was performed, already handled above
            pass
        elif args.files:
            # Proceed with analysis
            analyzer = LicenseAnalyzer(
                spdx_dir=args.spdx_dir, # Use the potentially custom or default SPDX dir
                cache_dir=args.cache_dir, # Use the potentially custom or default DB cache dir
                embedding_model_name=args.embedding_model
            )

            # Analyze files
            if len(args.files) == 1:
                file_path = args.files[0]
                matches = analyzer.analyze_file(file_path, args.top_n)

                # Filter by minimum score
                matches = [m for m in matches if m.score >= args.min_score]

                results = {file_path: matches}
            else:
                results = analyzer.analyze_multiple_files(args.files, args.top_n)

                # Filter by minimum score
                for file_path in results:
                    results[file_path] = [m for m in results[file_path] if m.score >= args.min_score]

            # Format and output results
            if args.format == "json":
                print(format_json_output(results))
            elif args.format == "csv":
                print(format_csv_output(results))
            else:  # text format
                for file_path, matches in results.items():
                    if len(results) > 1:
                        print()  # Add separator for multiple files
                    print(format_text_output(file_path, matches))

            # Show database stats if verbose
            if args.verbose:
                stats = analyzer.get_database_stats()
                print(f"\nDatabase stats: {stats['licenses']} licenses, "
                      f"{stats['exceptions']} exceptions ({stats['total']} total)",
                      file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# Example usage as a library:
"""
from license_analyzer.core import LicenseAnalyzer, analyze_license_file

# Method 1: Using the class directly
analyzer = LicenseAnalyzer() # Will use default cached SPDX data

# Analyze a single file
matches = analyzer.analyze_file("LICENSE.txt", top_n=10)
for match in matches:
    print(f"{match.name}: {match.score:.4f} ({match.method.value})")

# Analyze multiple files
results = analyzer.analyze_multiple_files(["LICENSE1.txt", "LICENSE2.txt"])
for file_path, matches in results.items():
    print(f"\n{file_path}:")
    for match in matches:
        print(f"  {match.name}: {match.score:.4f}")

# Analyze text directly
license_text = "MIT License\n\nCopyright (c) 2024..."
matches = analyzer.analyze_text(license_text)

# Method 2: Using convenience functions
matches = analyze_license_file("LICENSE.txt")
"""
