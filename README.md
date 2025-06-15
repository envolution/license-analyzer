# license-analyzer

**SPDX license identification using hashes, fingerprints, and semantic similarity.**  
Supports command-line usage as well as Python module integration.

---

## 📦 Installation

```bash
pip install license-analyzer
```

To install from source (e.g., for development):

```bash
git clone https://github.com/yourorg/license-analyzer.git
cd license-analyzer
pip install .
```

---

## 🚀 Command-Line Usage

Once installed, the CLI tool is available as:

```bash
license-analyzer [OPTIONS] FILE [FILE...]
```

### 🔧 Common Options

| Option | Description |
|--------|-------------|
| `--top-n N` | Return top N matches per file. If omitted, returns **all matches tied for highest score**. |
| `--format {text,json,csv}` | Output format. Default is `text`. |
| `--min-score FLOAT` | Filter out matches with a score below this threshold (default: `0.0`). |
| `--spdx-dir DIR` | Path to SPDX license text files. Defaults to `~/.cache/license-analyzer/spdx/text`. |
| `--cache-dir DIR` | Path to cache directory for license database. |
| `--embedding-model NAME` | SentenceTransformer model (default: `all-MiniLM-L6-v2`). |
| `--update, -u` | Force update of SPDX license data from GitHub. |
| `--verbose, -v` | Show progress and debug logs. |

### 📄 Examples

#### Basic usage

```bash
license-analyzer LICENSE
```

#### Multiple files

```bash
license-analyzer license1.txt license2.txt
```

#### JSON output with top 3 matches

```bash
license-analyzer --format json --top-n 3 LICENSE
```

#### Force SPDX update

```bash
license-analyzer --update
```

---

## 🐍 Python Module Usage

You can also use `license-analyzer` directly in your Python code:

```python
from license_analyzer.core import LicenseAnalyzer

analyzer = LicenseAnalyzer()
matches = analyzer.analyze_file("LICENSE")

for match in matches:
    print(match.name, match.score, match.method)
```

Or, if you want to analyze text (rather than a file):

```python
text = open("LICENSE").read()
matches = analyzer.analyze_text(text)

for match in matches:
    print(match.name, match.score, match.method)
```

Use `top_n=None` to get all tied top-scoring matches:

```python
matches = analyzer.analyze_text(text, top_n=None)
```

---

## 📈 Output Format (CLI)

### Text (default)

```text
Analysis results for: LICENSE
------------------------------------------------------------
MIT                            score: 1.0000  method: sha256
```

### JSON

```json
{
  "LICENSE": [
    {
      "name": "MIT",
      "score": 1.0,
      "method": "sha256"
    }
  ]
}
```

### CSV

```csv
file_path,license_name,score,method
"LICENSE","MIT",1.0,"sha256"
```

---

## 🔄 Updating SPDX License Data

By default, license data is stored under:

```
~/.cache/license-analyzer/spdx
```

To update the SPDX license texts (from GitHub):

```bash
license-analyzer --update
```

This refreshes cached licenses and triggers database rebuild if needed.

---

## 🧠 Matching Strategies

- ✅ SHA256 Hash Match
- ✅ Canonical Fingerprint Match
- ✅ Semantic Embedding Match (via [sentence-transformers](https://www.sbert.net/))

---

## 📝 License

SPDX-License-Identifier: Apache-2.0

---
