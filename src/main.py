"""
License Analyzer Module

A robust Python module for analyzing and comparing software licenses using
multiple matching strategies including SHA256, fingerprinting, and semantic embeddings.
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class MatchMethod(Enum):
    """Enumeration of available matching methods."""
    SHA256 = "sha256"
    FINGERPRINT = "fingerprint"
    EMBEDDING = "embedding"


@dataclass
class LicenseMatch:
    """Represents a license match result."""
    name: str
    score: float
    method: MatchMethod
    license_type: str = "license"  # "license" or "exception"
    
    def __post_init__(self):
        """Validate score range."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")


@dataclass
class DatabaseEntry:
    """Represents a license database entry."""
    name: str
    sha256: str
    fingerprint: str
    embedding: Optional[List[float]] = None
    updated: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    file_path: Optional[Path] = None


class LicenseDatabase:
    """Manages the license database with lazy loading and caching."""
    
    def __init__(self, spdx_dir: Path, cache_dir: Path, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.spdx_dir = Path(spdx_dir)
        self.exceptions_dir = self.spdx_dir / "exceptions"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None
        
        self.licenses_db_path = self.cache_dir / "licenses.json"
        self.exceptions_db_path = self.cache_dir / "exceptions.json"
        
        self._licenses_db: Optional[Dict[str, DatabaseEntry]] = None
        self._exceptions_db: Optional[Dict[str, DatabaseEntry]] = None
        
        self.logger = logging.getLogger(__name__)
    
    @property
    def embedding_model(self):
        """Lazy load the embedding model only when needed."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except ImportError:
                raise ImportError("sentence-transformers is required for embedding-based matching")
        return self._embedding_model
    
    def _sha256sum(self, path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    
    def _sha256sum_text(self, text: str) -> str:
        """Calculate SHA256 hash of text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing."""
        return " ".join(text.lower().split())
    
    def _canonical_fingerprint(self, text: str) -> str:
        """Generate canonical fingerprint from text."""
        tokens = sorted(set(self._normalize_text(text).split()))
        return hashlib.sha256(" ".join(tokens).encode('utf-8')).hexdigest()
    
    def _load_existing_db(self, db_path: Path) -> Dict[str, dict]:
        """Load existing database from JSON file."""
        if not db_path.exists():
            return {}
        
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"Failed to load database {db_path}: {e}")
            return {}
    
    def _save_db(self, db: Dict[str, dict], db_path: Path) -> None:
        """Save database to JSON file."""
        try:
            with open(db_path, 'w', encoding='utf-8') as f:
                json.dump(db, f, indent=2, ensure_ascii=False)
        except IOError as e:
            self.logger.error(f"Failed to save database {db_path}: {e}")
            raise
    
    def _update_database(self, source_dir: Path, db_path: Path, db_type: str) -> Dict[str, DatabaseEntry]:
        """Update database from source directory."""
        if not source_dir.exists():
            self.logger.warning(f"Source directory does not exist: {source_dir}")
            return {}
        
        raw_db = self._load_existing_db(db_path)
        db = {}
        updated = False
        
        for file_path in sorted(source_dir.glob("*.txt")):
            name = file_path.name
            current_sha = self._sha256sum(file_path)
            
            # Check if file needs updating
            if name in raw_db and raw_db[name].get("sha256") == current_sha:
                # File unchanged, convert to DatabaseEntry
                entry_data = raw_db[name]
                db[name] = DatabaseEntry(
                    name=name,
                    sha256=entry_data["sha256"],
                    fingerprint=entry_data["fingerprint"],
                    embedding=entry_data.get("embedding"),
                    updated=entry_data["updated"],
                    file_path=file_path
                )
                continue
            
            # File is new or changed
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                fingerprint = self._canonical_fingerprint(text)
                
                db[name] = DatabaseEntry(
                    name=name,
                    sha256=current_sha,
                    fingerprint=fingerprint,
                    embedding=None,  # Will be computed on demand
                    file_path=file_path
                )
                
                # Update raw database
                raw_db[name] = {
                    "sha256": current_sha,
                    "fingerprint": fingerprint,
                    "embedding": None,  # Placeholder, computed on demand
                    "updated": datetime.now(UTC).isoformat()
                }
                
                updated = True
                self.logger.info(f"Updated {db_type}: {name}")
                
            except IOError as e:
                self.logger.error(f"Failed to read {file_path}: {e}")
                continue
        
        if updated:
            self._save_db(raw_db, db_path)
            self.logger.info(f"Updated {db_type} database: {db_path}")
        
        return db
    
    def _get_embedding(self, entry: DatabaseEntry) -> np.ndarray:
        """Get embedding for a database entry, computing if necessary."""
        if entry.embedding is not None:
            return np.array(entry.embedding, dtype=np.float32)
        
        # Need to compute embedding
        if entry.file_path and entry.file_path.exists():
            try:
                with open(entry.file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                embedding = self.embedding_model.encode(text)
                entry.embedding = embedding.tolist()
                
                # Update the database file
                self._update_embedding_in_db(entry)
                
                return embedding
            except IOError as e:
                self.logger.error(f"Failed to read file for embedding: {entry.file_path}: {e}")
                raise
        else:
            raise ValueError(f"Cannot compute embedding for {entry.name}: file not found")
    
    def _update_embedding_in_db(self, entry: DatabaseEntry) -> None:
        """Update embedding in the database file."""
        # Determine which database this entry belongs to
        db_path = self.licenses_db_path
        if entry.file_path and "exceptions" in str(entry.file_path):
            db_path = self.exceptions_db_path
        
        try:
            raw_db = self._load_existing_db(db_path)
            if entry.name in raw_db:
                raw_db[entry.name]["embedding"] = entry.embedding
                raw_db[entry.name]["updated"] = datetime.now(UTC).isoformat()
                self._save_db(raw_db, db_path)
        except Exception as e:
            self.logger.warning(f"Failed to update embedding in database: {e}")
    
    @property
    def licenses_db(self) -> Dict[str, DatabaseEntry]:
        """Get licenses database, updating if necessary."""
        if self._licenses_db is None:
            self._licenses_db = self._update_database(
                self.spdx_dir, self.licenses_db_path, "licenses"
            )
        return self._licenses_db
    
    @property
    def exceptions_db(self) -> Dict[str, DatabaseEntry]:
        """Get exceptions database, updating if necessary."""
        if self._exceptions_db is None:
            self._exceptions_db = self._update_database(
                self.exceptions_dir, self.exceptions_db_path, "exceptions"
            )
        return self._exceptions_db
    
    def get_all_entries(self) -> Dict[str, Tuple[DatabaseEntry, str]]:
        """Get all database entries with their types."""
        all_entries = {}
        
        for name, entry in self.licenses_db.items():
            all_entries[name] = (entry, "license")
        
        for name, entry in self.exceptions_db.items():
            all_entries[name] = (entry, "exception")
        
        return all_entries


class LicenseAnalyzer:
    """Main license analyzer class."""
    
    def __init__(self, spdx_dir: Union[str, Path] = "/usr/share/licenses/spdx",
                 cache_dir: Optional[Union[str, Path]] = None,
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the license analyzer.
        
        Args:
            spdx_dir: Path to SPDX licenses directory
            cache_dir: Path to cache directory (default: ~/.cache/spdx)
            embedding_model_name: Name of the sentence transformer model
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "spdx"
        
        self.db = LicenseDatabase(spdx_dir, cache_dir, embedding_model_name)
        self.logger = logging.getLogger(__name__)
    
    def analyze_file(self, file_path: Union[str, Path], top_n: int = 5) -> List[LicenseMatch]:
        """
        Analyze a single license file.
        
        Args:
            file_path: Path to the license file to analyze
            top_n: Number of top matches to return
            
        Returns:
            List of LicenseMatch objects sorted by score (descending)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"License file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except IOError as e:
            raise IOError(f"Failed to read license file {file_path}: {e}")
        
        return self.analyze_text(text, top_n)
    
    def analyze_text(self, text: str, top_n: int = 5) -> List[LicenseMatch]:
        """
        Analyze license text.
        
        Args:
            text: License text to analyze
            top_n: Number of top matches to return
            
        Returns:
            List of LicenseMatch objects sorted by score (descending)
        """
        if not text.strip():
            return []
        
        text_sha = self.db._sha256sum_text(text)
        text_fingerprint = self.db._canonical_fingerprint(text)
        
        all_entries = self.db.get_all_entries()
        
        sha_matches = []
        fingerprint_matches = []
        embedding_matches = []
        
        # Check for exact matches first
        for name, (entry, entry_type) in all_entries.items():
            if text_sha == entry.sha256:
                sha_matches.append(LicenseMatch(
                    name=name, score=1.0, method=MatchMethod.SHA256, license_type=entry_type
                ))
            elif text_fingerprint == entry.fingerprint:
                fingerprint_matches.append(LicenseMatch(
                    name=name, score=1.0, method=MatchMethod.FINGERPRINT, license_type=entry_type
                ))
        
        # If we have perfect matches, return them but also check for other perfect matches
        if sha_matches or fingerprint_matches:
            perfect_matches = sha_matches + fingerprint_matches
            if len(perfect_matches) >= top_n:
                return perfect_matches[:top_n]
            # Continue to find more matches up to top_n
            remaining = top_n - len(perfect_matches)
        else:
            perfect_matches = []
            remaining = top_n
        
        # Only compute embeddings if we need more matches
        if remaining > 0:
            try:
                from sentence_transformers import util
                
                text_embedding = self.db.embedding_model.encode(text)
                
                for name, (entry, entry_type) in all_entries.items():
                    # Skip if already in perfect matches
                    if any(match.name == name for match in perfect_matches):
                        continue
                    
                    try:
                        entry_embedding = self.db._get_embedding(entry)
                        similarity = float(util.cos_sim(text_embedding, entry_embedding)[0][0])
                        
                        embedding_matches.append(LicenseMatch(
                            name=name, score=similarity, method=MatchMethod.EMBEDDING, license_type=entry_type
                        ))
                    except Exception as e:
                        self.logger.warning(f"Failed to compute embedding similarity for {name}: {e}")
                        continue
                
                # Sort embedding matches by score
                embedding_matches.sort(key=lambda x: x.score, reverse=True)
                
            except ImportError:
                self.logger.warning("sentence-transformers not available, skipping embedding analysis")
        
        # Combine all matches
        all_matches = perfect_matches + embedding_matches[:remaining]
        
        # Sort by score (perfect matches first, then by similarity)
        all_matches.sort(key=lambda x: (x.score, x.method.value == "sha256", x.method.value == "fingerprint"), reverse=True)
        
        return all_matches[:top_n]
    
    def analyze_multiple_files(self, file_paths: List[Union[str, Path]], 
                             top_n: int = 5) -> Dict[str, List[LicenseMatch]]:
        """
        Analyze multiple license files.
        
        Args:
            file_paths: List of paths to license files
            top_n: Number of top matches to return per file
            
        Returns:
            Dictionary mapping file paths to lists of LicenseMatch objects
        """
        results = {}
        
        for file_path in file_paths:
            file_path = Path(file_path)
            try:
                matches = self.analyze_file(file_path, top_n)
                results[str(file_path)] = matches
            except Exception as e:
                self.logger.error(f"Failed to analyze {file_path}: {e}")
                results[str(file_path)] = []
        
        return results
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get statistics about the license database."""
        return {
            "licenses": len(self.db.licenses_db),
            "exceptions": len(self.db.exceptions_db),
            "total": len(self.db.licenses_db) + len(self.db.exceptions_db)
        }


# Convenience functions for backward compatibility
def analyze_license_file(file_path: Union[str, Path], top_n: int = 5, 
                        spdx_dir: Union[str, Path] = "/usr/share/licenses/spdx") -> List[LicenseMatch]:
    """
    Convenience function to analyze a single license file.
    
    Args:
        file_path: Path to the license file
        top_n: Number of top matches to return
        spdx_dir: Path to SPDX licenses directory
        
    Returns:
        List of LicenseMatch objects
    """
    analyzer = LicenseAnalyzer(spdx_dir=spdx_dir)
    return analyzer.analyze_file(file_path, top_n)


def analyze_license_text(text: str, top_n: int = 5,
                        spdx_dir: Union[str, Path] = "/usr/share/licenses/spdx") -> List[LicenseMatch]:
    """
    Convenience function to analyze license text.
    
    Args:
        text: License text to analyze
        top_n: Number of top matches to return
        spdx_dir: Path to SPDX licenses directory
        
    Returns:
        List of LicenseMatch objects
    """
    analyzer = LicenseAnalyzer(spdx_dir=spdx_dir)
    return analyzer.analyze_text(text, top_n)


if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if len(sys.argv) < 2:
        print("Usage: python license_analyzer.py <license-file> [<license-file2> ...]")
        sys.exit(1)
    
    analyzer = LicenseAnalyzer()
    
    if len(sys.argv) == 2:
        # Single file analysis
        file_path = sys.argv[1]
        matches = analyzer.analyze_file(file_path)
        
        print(f"Analysis results for: {file_path}")
        print("=" * 60)
        
        if matches:
            for match in matches:
                print(f"{match.name:<30} score: {match.score:.4f}  method: {match.method.value}  type: {match.license_type}")
        else:
            print("No matches found.")
    else:
        # Multiple file analysis
        file_paths = sys.argv[1:]
        results = analyzer.analyze_multiple_files(file_paths)
        
        for file_path, matches in results.items():
            print(f"\nAnalysis results for: {file_path}")
            print("=" * 60)
            
            if matches:
                for match in matches:
                    print(f"{match.name:<30} score: {match.score:.4f}  method: {match.method.value}  type: {match.license_type}")
            else:
                print("No matches found.")
    
    # Show database stats
    stats = analyzer.get_database_stats()
    print(f"\nDatabase contains {stats['licenses']} licenses and {stats['exceptions']} exceptions ({stats['total']} total)")
