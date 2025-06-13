# License Analyzer

A robust Python module for analyzing and comparing software licenses using multiple matching strategies including SHA256, fingerprinting, and semantic embeddings.

## Features

- **Multiple Matching Methods**: SHA256 exact matching, canonical fingerprinting, and semantic embeddings
- **Lazy Loading**: Embedding models are only loaded when needed for performance
- **Efficient Caching**: Automatically manages license database with incremental updates
- **Batch Processing**: Analyze multiple license files in a single operation
- **Flexible API**: Both object-oriented and functional interfaces
- **Command Line Interface**: Easy-to-use CLI for quick analysis
- **Comprehensive Testing**: Full test suite with mocking for reliable operation

## Installation

```bash
# Install from PyPI (when published)
pip install license-analyzer

# Or install from source
git clone https://github.com/yourusername/license-analyzer.git
cd license-analyzer
pip install -e .

# For development
pip install -e .[dev]
```

## Requirements

- Python 3.8+
- numpy
- sentence-transformers (for semantic analysis)
- torch (dependency of sentence-transformers)

## Quick Start

### As a Library

```python
from license_analyzer import LicenseAnalyzer

# Initialize analyzer
analyzer = LicenseAnalyzer()

# Analyze a single file
matches = analyzer.analyze_file("LICENSE.txt", top_n=5)
for match in matches:
    print(f"{match.name}: {match.score:.4f} ({match.method.value})")

# Analyze text directly
license_text = """MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted..."""

matches = analyzer.analyze_text(license_text)
print(f"Best match: {matches[0].name} (score: {matches[0].score:.4f})")

# Analyze multiple files
results = analyzer.analyze_multiple_files([
    "LICENSE1.txt", 
    "LICENSE2.txt", 
    "LICENSE3.txt"
])

for file_path, matches in results.items():
    print(f"\n{file_path}:")
    for match in matches[:3]:  # Top 3 matches
        print(f"  {match.name}: {match.score:.4f}")
```

### Command Line Interface

```bash
# Analyze a single file
license-analyzer LICENSE.txt

# Analyze multiple files
license-analyzer LICENSE1.txt LICENSE2.txt LICENSE3.txt

# Get top 10 matches in JSON format
license-analyzer --format json --top-n 10 LICENSE.txt

# Set minimum score threshold
license-analyzer --min-score 0.8 LICENSE.txt

# Use custom SPDX directory
license-analyzer --spdx-dir /custom/path/spdx LICENSE.txt

# Verbose output with database stats
license-analyzer --verbose LICENSE.txt
```

## Architecture

### Core Components

1. **LicenseAnalyzer**: Main interface class
2. **LicenseDatabase**: Manages license data with lazy loading
3. **LicenseMatch**: Represents analysis results
4. **DatabaseEntry**: Internal license database entry
5. **MatchMethod**: Enumeration of matching strategies

### Matching Strategies

1. **SHA256**: Exact byte-for-byte matching (fastest, most accurate)
2. **Fingerprint**: Canonical token-based matching (handles formatting differences)
3. **Embedding**: Semantic similarity using sentence transformers (most flexible)

### Performance Optimization

- **Lazy Loading**: Embedding models are only loaded when needed
- **Incremental Updates**: Database only processes changed files
- **Smart Matching**: Perfect matches skip expensive embedding computation
- **Efficient Caching**: JSON-based database with SHA
