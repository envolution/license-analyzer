#!/usr/bin/env python3
"""
Command Line Interface for License Analyzer

Example usage:
    python cli.py single_license.txt
    python cli.py license1.txt license2.txt license3.txt
    python cli.py --top-n 10 --format json license.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Assuming the license_analyzer module is in the same directory or installed
from license_analyzer import LicenseAnalyzer, LicenseMatch


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
  %(prog)s --spdx-dir /custom/spdx/path LICENSE
        """
    )
    
    parser.add_argument(
        "files",
        nargs="+",
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
        default="/usr/share/licenses/spdx",
        help="Path to SPDX licenses directory (default: /usr/share/licenses/spdx)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Path to cache directory (default: ~/.cache/spdx)"
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
    
    args = parser.parse_args()
    
    # Set up logging
    import logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize analyzer
        analyzer = LicenseAnalyzer(
            spdx_dir=args.spdx_dir,
            cache_dir=args.cache_dir,
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


if __name__ == "__main__":
    main()


# Example usage as a library:
"""
from license_analyzer import LicenseAnalyzer, analyze_license_file

# Method 1: Using the class directly
analyzer = LicenseAnalyzer()

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
