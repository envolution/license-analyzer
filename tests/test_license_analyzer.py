import unittest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
from datetime import datetime, UTC
from appdirs import user_cache_dir # Import for testing default cache_dir

# Import the module components
from license_analyzer.core import (
    LicenseAnalyzer,
    LicenseDatabase,
    LicenseMatch,
    DatabaseEntry,
    MatchMethod,
    analyze_license_file,
    analyze_license_text,
)
from license_analyzer.updater import LicenseUpdater # Potentially useful for full integration tests

# Suppress logging output from the library during tests for cleaner test output
import logging
logging.getLogger("license_analyzer").setLevel(logging.CRITICAL)


class TestLicenseMatch(unittest.TestCase):
    """Test the LicenseMatch dataclass."""

    def test_valid_match(self):
        """Test creating a valid LicenseMatch."""
        match = LicenseMatch(
            name="MIT",
            score=0.95,
            method=MatchMethod.EMBEDDING,
        )
        self.assertEqual(match.name, "MIT")
        self.assertEqual(match.score, 0.95)
        self.assertEqual(match.method, MatchMethod.EMBEDDING)

    def test_invalid_score_low(self):
        """Test that scores below 0.0 raise ValueError."""
        with self.assertRaises(ValueError):
            LicenseMatch(name="test", score=-0.1, method=MatchMethod.SHA256)

    def test_invalid_score_high(self):
        """Test that scores above 1.0 raise ValueError."""
        with self.assertRaises(ValueError):
            LicenseMatch(name="test", score=1.1, method=MatchMethod.SHA256)


class TestDatabaseEntry(unittest.TestCase):
    """Test the DatabaseEntry dataclass."""

    def test_database_entry_creation(self):
        """Test creating a DatabaseEntry."""
        # The 'updated' field has a default_factory, so we don't need to pass it unless specific
        entry = DatabaseEntry(
            name="MIT",
            sha256="abcd1234",
            fingerprint="efgh5678",
            embedding=[0.1, 0.2, 0.3],
            file_path=Path("/test/MIT.txt"),
        )
        self.assertEqual(entry.name, "MIT")
        self.assertEqual(entry.sha256, "abcd1234")
        self.assertEqual(entry.fingerprint, "efgh5678")
        self.assertEqual(entry.embedding, [0.1, 0.2, 0.3])
        self.assertEqual(entry.file_path, Path("/test/MIT.txt"))
        self.assertIsInstance(datetime.fromisoformat(entry.updated), datetime) # Check default 'updated'
        self.assertTrue(entry.updated.endswith('+00:00') or entry.updated.endswith('Z')) # Check UTC


class TestLicenseDatabase(unittest.TestCase):
    """Test the LicenseDatabase class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.spdx_dir = Path(self.temp_dir) / "spdx"
        # Consistent with core.py's default sub-directory for db cache
        self.cache_dir = Path(self.temp_dir) / "db_cache" 
        self.licenses_db_path = self.cache_dir / "licenses.json"

        # Create directory structure
        self.spdx_dir.mkdir(parents=True)
        self.cache_dir.mkdir(parents=True)

        self.mit_content = """MIT License

Copyright (c) 2024 Test

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

        # License files are stored *without* .txt suffix by updater.py now
        (self.spdx_dir / "MIT").write_text(self.mit_content)

        self.db = LicenseDatabase(self.spdx_dir, self.cache_dir, "all-MiniLM-L6-v2")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_text_processing(self):
        """Test text normalization and fingerprinting."""
        text = "  Hello   WORLD  \n  Test  "
        normalized = self.db._normalize_text(text)
        self.assertEqual(normalized, "hello world test")

        fingerprint = self.db._canonical_fingerprint(text)
        self.assertIsInstance(fingerprint, str)
        self.assertEqual(len(fingerprint), 64)  # SHA256 hex length

    def test_sha256_calculation(self):
        """Test SHA256 calculation."""
        test_file = self.spdx_dir / "test_file_for_sha.txt" # New dummy file for sha test
        test_file.write_text("test content")

        sha = self.db._sha256sum(test_file)
        self.assertIsInstance(sha, str)
        self.assertEqual(len(sha), 64)
        # Corrected SHA for "test content"
        self.assertEqual(sha, "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72")

        # Test text SHA256
        text_sha = self.db._sha256sum_text("test content")
        self.assertEqual(sha, text_sha)

    @patch("sentence_transformers.SentenceTransformer")
    def test_embedding_model_lazy_loading(self, mock_transformer):
        """Test that embedding model is loaded lazily."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        # Model should not be loaded initially
        self.assertIsNone(self.db._embedding_model)

        # Access should trigger loading
        model = self.db.embedding_model
        self.assertIsNotNone(model)
        mock_transformer.assert_called_once_with("all-MiniLM-L6-v2")

        self.assertEqual(model, mock_model)

        self.assertIsNotNone(self.db._embedding_model)

    @patch.object(LicenseDatabase, '_save_db') # Prevent actual file writes during this test
    def test_update_database_initial_load(self, mock_save_db):
        """Test initial loading and processing of licenses into the database."""
        # Ensure the _licenses_db is initially None for this test to trigger update
        self.db._licenses_db = None
        mock_progress_callback = Mock()
        
        # Trigger the internal update method directly
        licenses_db_entries = self.db._update_database(
            self.spdx_dir, self.licenses_db_path, "licenses", mock_progress_callback
        )
        
        self.assertIn("MIT", licenses_db_entries)
        mit_entry = licenses_db_entries["MIT"]
        self.assertIsInstance(mit_entry, DatabaseEntry)
        self.assertEqual(mit_entry.name, "MIT")
        self.assertIsNotNone(mit_entry.sha256)
        self.assertIsNotNone(mit_entry.fingerprint)
        self.assertEqual(mit_entry.file_path, self.spdx_dir / "MIT") # Check file_path
        self.assertIsNone(mit_entry.embedding) # Embedding is not computed on initial load

        # Verify progress callback was called
        mock_progress_callback.assert_any_call(0, 1, "Processing licenses: MIT") # Initial call for MIT
        mock_progress_callback.assert_any_call(1, 1, "Finished licenses database update.") # Final call

        # Ensure save was called
        mock_save_db.assert_called_once()
        saved_db_data = mock_save_db.call_args[0][0]
        self.assertIn("MIT", saved_db_data)
        self.assertIsNone(saved_db_data["MIT"]["embedding"]) # Verify embedding is null in saved data

    @patch("sentence_transformers.SentenceTransformer")
    @patch.object(LicenseDatabase, '_save_db')
    def test_get_embedding_computation_and_cache_update(self, mock_save_db, mock_transformer):
        """Test _get_embedding computes and caches embedding."""
        mock_model = Mock()
        # Encode returns a 1D array for a single input text
        mock_model.encode.return_value = np.array([0.5, 0.6, 0.7], dtype=np.float32) 
        mock_transformer.return_value = mock_model

        # --- FIX: Manually create the licenses.json file as it would be after initial _update_database ---
        initial_db_content = {
            "MIT": {
                "sha256": self.db._sha256sum_text(self.mit_content),
                "fingerprint": self.db._canonical_fingerprint(self.mit_content),
                "embedding": None,
                "updated": datetime.now(UTC).isoformat()
            }
        }
        with open(self.licenses_db_path, "w", encoding="utf-8") as f:
            json.dump(initial_db_content, f, indent=2, ensure_ascii=False)
        # --- END FIX ---

        # Pre-populate the _licenses_db in memory (optional, but good practice for clarity)
        # Load from the file we just created, as _load_existing_db would do
        raw_db_from_file = self.db._load_existing_db(self.licenses_db_path)
        self.db._licenses_db = {
            name: DatabaseEntry(name=name, file_path=self.spdx_dir / name, **data)
            for name, data in raw_db_from_file.items()
        }

        mit_entry = self.db._licenses_db["MIT"]
        
        # Call _get_embedding, which should trigger computation and internal update
        embedding = self.db._get_embedding(mit_entry)
        
        # Verify embedding was computed
        # Assert against a 1D array, and use almost_equal for floats
        np.testing.assert_array_almost_equal(embedding, np.array([0.5, 0.6, 0.7]))
        # The stored embedding is the result of .tolist(), so it's a Python list
        np.testing.assert_array_almost_equal(np.array(mit_entry.embedding), np.array([0.5, 0.6, 0.7]))

        # Verify SentenceTransformer.encode was called
        mock_model.encode.assert_called_once_with(self.mit_content)

        # Verify _save_db was called to persist the embedding (THIS ASSERTION NOW PASSES)
        mock_save_db.assert_called_once()
        saved_db_data = mock_save_db.call_args[0][0]
        self.assertIn("MIT", saved_db_data)
        # Verify stored embedding in saved data (it will be a list of floats)
        np.testing.assert_array_almost_equal(np.array(saved_db_data["MIT"]["embedding"]), np.array([0.5, 0.6, 0.7]))


class TestLicenseAnalyzer(unittest.TestCase):
    """Test the main LicenseAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.spdx_dir = Path(self.temp_dir) / "spdx"
        self.cache_dir = Path(self.temp_dir) / "db_cache"

        self.spdx_dir.mkdir(parents=True)
        self.cache_dir.mkdir(parents=True)

        self.mit_content = "MIT License\n\nCopyright (c) 2024"
        self.apache_content = "Apache License Version 2.0, January 2004"
        self.gpl_content = "GNU General Public License v3.0"

        # Write license files as updater would (no .txt suffix in spdx_dir)
        (self.spdx_dir / "MIT").write_text(self.mit_content)
        (self.spdx_dir / "Apache-2.0").write_text(self.apache_content)
        (self.spdx_dir / "GPL-3.0").write_text(self.gpl_content)

        # Patch the sentence transformer and util for similarity calculations
        self.patcher_transformer = patch("sentence_transformers.SentenceTransformer")
        self.patcher_util = patch("sentence_transformers.util")

        self.mock_transformer_class = self.patcher_transformer.start()
        self.mock_util = self.patcher_util.start()

        self.mock_model = Mock()
        # Ensure encode returns a numpy array with shape (1, embedding_dim)
        self.mock_model.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32) # 1D array
        self.mock_transformer_class.return_value = self.mock_model

        # Ensure cos_sim returns a 2D numpy array
        self.mock_util.cos_sim.return_value = np.array([[0.8]])

        # Mock the progress callbacks used in Analyzer's __init__
        self.mock_db_progress_callback = Mock()
        self.analyzer = LicenseAnalyzer(
            spdx_dir=self.spdx_dir,
            cache_dir=self.cache_dir,
            db_progress_callback=self.mock_db_progress_callback,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        self.patcher_transformer.stop()
        self.patcher_util.stop()
        shutil.rmtree(self.temp_dir)

    def test_analyzer_initialization(self):
        """Test LicenseAnalyzer initializes and updates database correctly."""
        # Check that the database update was triggered in __init__
        self.mock_db_progress_callback.assert_called()
        self.assertEqual(len(self.analyzer.db.licenses_db), 3) # MIT, Apache-2.0, GPL-3.0

        # Verify encode was NOT called during analyzer init, as _get_embedding is lazy and not yet triggered
        self.assertEqual(self.mock_model.encode.call_count, 0)
        
        # Access one of the database entries to ensure it exists and has correct info
        mit_entry = self.analyzer.db.licenses_db["MIT"]
        self.assertEqual(mit_entry.name, "MIT")
        self.assertEqual(mit_entry.file_path, self.spdx_dir / "MIT")


    @patch.object(LicenseDatabase, '_get_embedding') # Mock this method to control embedding retrieval
    def test_analyze_text_exact_sha_match(self, mock_get_embedding):
        """Test analyzing text with exact SHA256 match."""
        # Setup mock_get_embedding to return a dummy array if called (shouldn't be for SHA match)
        mock_get_embedding.return_value = np.array([0.9, 0.8, 0.7]) # 1D array

        # Analyze text that is an exact SHA256 match
        # Using top_n=1 to ensure embedding calculation for *other* DB entries is skipped
        matches = self.analyzer.analyze_text(self.mit_content, top_n=1) 

        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].name, "MIT")
        self.assertEqual(matches[0].score, 1.0)
        self.assertEqual(matches[0].method, MatchMethod.SHA256)
        
        # Should not call encode on the input text if a perfect SHA match is found
        self.mock_model.encode.assert_not_called() 
        # Should not call _get_embedding for database entries either because top_n=1 and exact match found
        mock_get_embedding.assert_not_called()


    @patch.object(LicenseDatabase, '_get_embedding')
    def test_analyze_text_exact_fingerprint_match(self, mock_get_embedding):
        """Test analyzing text with exact fingerprint match (but not SHA256)."""
        # Create content that has the same canonical fingerprint as MIT but different SHA
        # This is achieved by changing whitespace/case which _normalize_text would fix for fingerprint,
        # but not for SHA256.
        fingerprint_only_content = self.mit_content.replace("Copyright (c) 2024", "copyright (C) 2024")
        
        # Verify it has different SHA but same fingerprint as original MIT content
        original_mit_sha = self.analyzer.db._sha256sum_text(self.mit_content)
        original_mit_fp = self.analyzer.db._canonical_fingerprint(self.mit_content)
        
        modified_sha = self.analyzer.db._sha256sum_text(fingerprint_only_content)
        modified_fp = self.analyzer.db._canonical_fingerprint(fingerprint_only_content)
        
        self.assertNotEqual(original_mit_sha, modified_sha)
        self.assertEqual(original_mit_fp, modified_fp)

        # Perform analysis. It should find the fingerprint match.
        # Using top_n=1 to ensure embedding calculation is skipped if a perfect match is found
        matches = self.analyzer.analyze_text(fingerprint_only_content, top_n=1)

        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].name, "MIT")
        self.assertEqual(matches[0].score, 1.0)
        self.assertEqual(matches[0].method, MatchMethod.FINGERPRINT)
        
        self.mock_model.encode.assert_not_called()
        mock_get_embedding.assert_not_called()


    @patch.object(LicenseDatabase, '_get_embedding')
    def test_analyze_text_embedding_match(self, mock_get_embedding):
        """Test analyzing text with embedding match when no exact match."""
        # Make a text that is not an exact SHA or FP match
        non_exact_text = "This is a license that is similar to MIT but not identical."
        
        # Configure the mock SentenceTransformer to encode the input text
        self.mock_model.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32) # 1D array
        # Configure cos_sim for the non-exact match
        self.mock_util.cos_sim.return_value = np.array([[0.75]])
        # Ensure _get_embedding is called and returns an embedding for existing licenses
        mock_get_embedding.return_value = np.array([0.9, 0.8, 0.7], dtype=np.float32) # 1D array

        # Test with a mock callback for per-entry embedding progress
        mock_per_entry_embed_callback = Mock()
        matches = self.analyzer.analyze_text(non_exact_text, per_entry_embed_callback=mock_per_entry_embed_callback)

        self.assertGreater(len(matches), 0)
        
        # Expect an embedding match
        embedding_matches = [m for m in matches if m.method == MatchMethod.EMBEDDING]
        self.assertGreater(len(embedding_matches), 0)
        # The specific name depends on the mock return order, so check if it's one of the known ones
        self.assertIn(embedding_matches[0].name, ["MIT", "Apache-2.0", "GPL-3.0"]) 
        self.assertEqual(embedding_matches[0].score, 0.75) # Mocked score

        # Verify that SentenceTransformer.encode was called for the input text
        self.mock_model.encode.assert_called_once_with(non_exact_text)
        # Verify that _get_embedding was called for each license in the DB
        self.assertEqual(mock_get_embedding.call_count, 3) # For MIT, Apache-2.0, GPL-3.0
        # Verify per_entry_embed_callback was called
        mock_per_entry_embed_callback.assert_called()


    def test_analyze_file_not_found(self):
        """Test analyzing non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.analyzer.analyze_file("non_existent_file.txt")

    def test_analyze_empty_text(self):
        """Test analyzing empty text."""
        matches = self.analyzer.analyze_text("")
        self.assertEqual(len(matches), 0)

        matches = self.analyzer.analyze_text("   \n  \t  ")
        self.assertEqual(len(matches), 0)

    @patch.object(LicenseDatabase, '_get_embedding')
    def test_analyze_multiple_files(self, mock_get_embedding):
        """Test analyzing multiple files with progress callback."""
        # Set up a dummy embedding for _get_embedding (called for DB entries)
        mock_get_embedding.return_value = np.array([0.9, 0.8, 0.7]) # 1D array

        # Configure mock_model.encode for the input file texts
        self.mock_model.encode.side_effect = [
            np.array([0.1, 0.2, 0.3]), # For file1 content
            np.array([0.4, 0.5, 0.6]), # For file2 content
        ]
        # Configure mock_util.cos_sim for comparisons
        # 3 DB licenses * 2 input files = 6 comparisons in total, each results in one cos_sim call
        self.mock_util.cos_sim.side_effect = [
            np.array([[0.95]]), # file1 vs MIT
            np.array([[0.85]]), # file1 vs Apache-2.0
            np.array([[0.75]]), # file1 vs GPL-3.0
            np.array([[0.9]]),  # file2 vs MIT
            np.array([[0.8]]),  # file2 vs Apache-2.0
            np.array([[0.7]]),  # file2 vs GPL-3.0
        ]

        # Create dummy input files
        file1_path = Path(self.temp_dir) / "input_license1.txt"
        file2_path = Path(self.temp_dir) / "input_license2.txt"
        file1_path.write_text("This is input license 1 content.")
        file2_path.write_text("This is input license 2 content.")

        mock_analysis_progress_callback = Mock()
        
        results = self.analyzer.analyze_multiple_files(
            [file1_path, file2_path],
            top_n=2,
            analysis_progress_callback=mock_analysis_progress_callback,
        )

        self.assertEqual(len(results), 2)
        self.assertIn(str(file1_path), results)
        self.assertIn(str(file2_path), results)

        # Verify calls to model.encode (once per input file)
        self.assertEqual(self.mock_model.encode.call_count, 2)
        self.mock_model.encode.assert_has_calls([
            call("This is input license 1 content."),
            call("This is input license 2 content."),
        ], any_order=False) # Order matters for side_effect setup

        # Verify calls to _get_embedding (for each DB license, for each input file)
        # 3 DB licenses * 2 input files = 6 calls to _get_embedding
        self.assertEqual(mock_get_embedding.call_count, 6) 

        # Verify analysis progress callback calls
        mock_analysis_progress_callback.assert_called()
        self.assertEqual(mock_analysis_progress_callback.call_args[0][1], 2) # Total files
        # Check messages for each file
        mock_analysis_progress_callback.assert_any_call(1, 2, f"Analyzing {file1_path.name}")
        mock_analysis_progress_callback.assert_any_call(2, 2, f"Analyzing {file2_path.name}")
        # Check final message
        mock_analysis_progress_callback.assert_any_call(2, 2, "Finished analyzing files.")

        # Check the results structure and content for one file
        file1_matches = results[str(file1_path)]
        self.assertEqual(len(file1_matches), 2) # top_n = 2
        # Expected highest score from side_effect for file1
        self.assertEqual(file1_matches[0].score, 0.95) 
        self.assertEqual(file1_matches[0].method, MatchMethod.EMBEDDING)


    def test_get_database_stats(self):
        """Test getting database statistics."""
        stats = self.analyzer.get_database_stats()
        self.assertIn("total_licenses", stats)
        self.assertEqual(stats["total_licenses"], 3) # MIT, Apache-2.0, GPL-3.0


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.spdx_dir = Path(self.temp_dir) / "spdx"
        # The default cache_dir used by LicenseAnalyzer when not explicitly passed
        # This needs to be correctly identified for the assertion
        self.default_cache_base_dir = Path(user_cache_dir(appname="license-analyzer", appauthor="envolution"))
        self.default_db_cache_dir = self.default_cache_base_dir / "db_cache" 
        self.default_spdx_data_dir = self.default_cache_base_dir / "spdx"

        self.spdx_dir.mkdir(parents=True)
        # Ensure default cache directories exist because LicenseAnalyzer's __init__ will try to create them
        self.default_db_cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_spdx_data_dir.mkdir(parents=True, exist_ok=True)

        self.mit_content = "MIT License\n\nTest content for convenience function"
        # Create license file without .txt suffix as updater would
        (self.spdx_dir / "MIT").write_text(self.mit_content) 

        # Mock SentenceTransformer and util globally for convenience functions
        # as they instantiate LicenseAnalyzer internally.
        self.patcher_transformer = patch("sentence_transformers.SentenceTransformer")
        self.patcher_util = patch("sentence_transformers.util")

        self.mock_transformer_class = self.patcher_transformer.start()
        self.mock_util = self.patcher_util.start()

        self.mock_model = Mock()
        self.mock_model.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32) # 1D array
        self.mock_transformer_class.return_value = self.mock_model
        self.mock_util.cos_sim.return_value = np.array([[0.95]])

        # Patch _get_embedding as convenience functions will trigger this during analysis
        self.patcher_get_embedding = patch("license_analyzer.core.LicenseDatabase._get_embedding")
        self.mock_get_embedding = self.patcher_get_embedding.start()
        self.mock_get_embedding.return_value = np.array([0.9, 0.8, 0.7], dtype=np.float32) # 1D array

    def tearDown(self):
        self.patcher_transformer.stop()
        self.patcher_util.stop()
        self.patcher_get_embedding.stop()
        # Clean up only the temp dir created for spdx files, not the default appdirs cache.
        shutil.rmtree(self.temp_dir)
        # Also clean up the default cache directories to ensure test isolation between runs
        if self.default_cache_base_dir.exists():
            shutil.rmtree(self.default_cache_base_dir)

    @patch("license_analyzer.core.LicenseAnalyzer") # Patch the class that convenience func instantiates
    def test_analyze_license_file_function(self, mock_analyzer_class):
        """Test analyze_license_file convenience function."""
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze_file.return_value = [
            LicenseMatch("MIT", 1.0, MatchMethod.SHA256) # Name is "MIT"
        ]
        mock_analyzer_class.return_value = mock_analyzer_instance

        test_file = Path(self.temp_dir) / "test_input_file.txt"
        test_file.write_text("test content")

        # The `spdx_dir` argument to `analyze_license_file` maps directly to `LicenseAnalyzer`'s `spdx_dir`.
        # Other arguments are NOT passed by the convenience function.
        matches = analyze_license_file(test_file, top_n=3, spdx_dir=self.spdx_dir)

        # Analyzer should be initialized with only the spdx_dir argument passed by the convenience func.
        # The other arguments default to None if not explicitly passed.
        mock_analyzer_class.assert_called_once_with(
            spdx_dir=self.spdx_dir, # This is the argument passed
            # No other arguments are passed by analyze_license_file, so don't assert them here
        )
        # analyze_file should be called with top_n, and per_entry_embed_callback is NOT passed by convenience func
        mock_analyzer_instance.analyze_file.assert_called_once_with(test_file, 3) # Removed per_entry_embed_callback=ANY
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].name, "MIT")

    @patch("license_analyzer.core.LicenseAnalyzer")
    def test_analyze_license_text_function(self, mock_analyzer_class):
        """Test analyze_license_text convenience function."""
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze_text.return_value = [
            LicenseMatch("MIT", 0.95, MatchMethod.EMBEDDING)
        ]
        mock_analyzer_class.return_value = mock_analyzer_instance

        test_text = "MIT License test"
        matches = analyze_license_text(test_text, top_n=5, spdx_dir=self.spdx_dir)

        # Analyzer should be initialized with only the spdx_dir argument passed by the convenience func.
        mock_analyzer_class.assert_called_once_with(
            spdx_dir=self.spdx_dir,
            # No other arguments are passed by analyze_license_text
        )
        mock_analyzer_instance.analyze_text.assert_called_once_with(test_text, 5) # Removed per_entry_embed_callback=ANY
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].name, "MIT")


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.spdx_dir = Path(self.temp_dir) / "spdx"
        self.cache_dir = Path(self.temp_dir) / "db_cache"

        self.spdx_dir.mkdir(parents=True)
        self.cache_dir.mkdir(parents=True)

        self.mit_license_content = """MIT License

Copyright (c) 2024 Test Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

        self.apache_license_content = """Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions."""
        
        # Ensure that the updater would put these files without .txt suffix
        (self.spdx_dir / "MIT").write_text(self.mit_license_content)
        (self.spdx_dir / "Apache-2.0").write_text(self.apache_license_content)

        # Patch the actual SentenceTransformer and util modules
        self.patcher_transformer = patch("sentence_transformers.SentenceTransformer")
        self.patcher_util = patch("sentence_transformers.util")

        self.mock_transformer_class = self.patcher_transformer.start()
        self.mock_util = self.patcher_util.start()

        self.mock_model = Mock()
        # Mock encode to return a 1D numpy array (for a single sentence)
        def mock_encode_side_effect(text):
            if "MIT License" in text:
                return np.array([0.1, 0.2, 0.3], dtype=np.float32)
            elif "Apache License" in text:
                return np.array([0.4, 0.5, 0.6], dtype=np.float32)
            else: # For any other content that might be encoded
                return np.array([0.7, 0.8, 0.9], dtype=np.float32)

        self.mock_model.encode.side_effect = mock_encode_side_effect
        
        self.mock_transformer_class.return_value = self.mock_model
        
        # Mock cos_sim to return a 2D numpy array for similarity calculation
        self.mock_util.cos_sim.return_value = np.array([[0.85]]) # Default similarity for non-exact matches

    def tearDown(self):
        self.patcher_transformer.stop()
        self.patcher_util.stop()
        shutil.rmtree(self.temp_dir)

    def test_full_workflow(self):
        """Test complete workflow from initialization to analysis."""
        mock_db_progress_callback = Mock()
        analyzer = LicenseAnalyzer(
            spdx_dir=self.spdx_dir,
            cache_dir=self.cache_dir,
            db_progress_callback=mock_db_progress_callback,
        )

        # Clear mocks after Analyzer init to only capture calls from analyze_text onwards
        self.mock_model.encode.reset_mock()
        self.mock_util.cos_sim.reset_mock()

        # Test exact match (should use SHA256)
        # Using top_n=1 to ensure embedding calculation for *other* DB entries is skipped
        matches = analyzer.analyze_text(self.mit_license_content, top_n=1)

        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].name, "MIT")
        self.assertEqual(matches[0].score, 1.0)
        self.assertEqual(matches[0].method, MatchMethod.SHA256)
        
        # Now, it truly should not have called encode or cos_sim for any reason
        # since top_n=1 and a perfect match was found.
        self.mock_model.encode.assert_not_called()
        self.mock_util.cos_sim.assert_not_called()
        
        # Reset mocks for next analysis phase
        self.mock_model.encode.reset_mock()
        self.mock_util.cos_sim.reset_mock()


        # Test similarity match (should use EMBEDDING)
        # This content is similar to MIT but not an exact SHA256/fingerprint match.
        similar_content = self.mit_license_content.replace("2024 Test Project", "2023 New Project")
        
        # We need to ensure that _get_embedding (called internally by analyze_text)
        # returns the pre-defined embeddings for the database entries.
        # This patch allows us to control the embeddings returned for the DB entries.
        with patch.object(LicenseDatabase, '_get_embedding') as mock_get_embedding:
            # Set up side_effect for mock_get_embedding to return 1D arrays
            # Corrected: Use a list for side_effect for sequential calls, each returns a 1D array
            mock_get_embedding.side_effect = [
                np.array([0.1, 0.2, 0.3], dtype=np.float32),  # For MIT
                np.array([0.4, 0.5, 0.6], dtype=np.float32)   # For Apache-2.0
            ]
            
            mock_per_entry_embed_callback = Mock() # For the analyze_text call
            matches = analyzer.analyze_text(similar_content, top_n=5, per_entry_embed_callback=mock_per_entry_embed_callback)

            self.assertGreater(len(matches), 0)
            
            # Expect an embedding match
            embedding_matches = [m for m in matches if m.method == MatchMethod.EMBEDDING]
            self.assertGreater(len(embedding_matches), 0)
            self.assertEqual(embedding_matches[0].score, 0.85) # Mocked score from cos_sim
            # The name can be either MIT or Apache-2.0 depending on internal iteration order, so check presence
            self.assertIn(embedding_matches[0].name, ["MIT", "Apache-2.0"]) 

            # Verify encode was called for the input text (once)
            self.mock_model.encode.assert_called_once_with(similar_content)
            # Verify _get_embedding was called for existing DB licenses (MIT and Apache-2.0)
            self.assertEqual(mock_get_embedding.call_count, 2)
            # Verify per_entry_embed_callback was called
            mock_per_entry_embed_callback.assert_called()


        # Test database stats
        stats = analyzer.get_database_stats()
        self.assertEqual(stats["total_licenses"], 2) # MIT and Apache-2.0 loaded


if __name__ == "__main__":
    unittest.main()
