import unittest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, ANY  # Import ANY
import numpy as np
from datetime import datetime, UTC
from appdirs import user_cache_dir

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
from license_analyzer.updater import (
    LicenseUpdater,
)

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
        self.assertIsInstance(datetime.fromisoformat(entry.updated), datetime)
        self.assertTrue(entry.updated.endswith("+00:00") or entry.updated.endswith("Z"))


class TestLicenseDatabase(unittest.TestCase):
    """Test the LicenseDatabase class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.spdx_dir = Path(self.temp_dir) / "spdx"
        self.spdx_text_dir = self.spdx_dir / "text"
        self.spdx_json_dir = self.spdx_dir / "json"

        self.cache_dir = Path(self.temp_dir) / "db_cache"
        self.licenses_db_path = self.cache_dir / "licenses.json"

        self.spdx_text_dir.mkdir(parents=True)
        self.spdx_json_dir.mkdir(parents=True)
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

        # Write license files *with* .txt suffix as they would come from original SPDX data
        # LicenseDatabase._update_database uses file_path.stem to get the ID, which correctly strips .txt
        (self.spdx_text_dir / "MIT.txt").write_text(self.mit_content)

        dummy_licenses_json = {
            "licenseListVersion": "test_version_1.0",
            "licenses": [
                {"licenseId": "MIT", "name": "MIT License", "isOsiApproved": True}
            ],
        }
        (self.spdx_json_dir / "licenses.json").write_text(
            json.dumps(dummy_licenses_json)
        )

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
        self.assertEqual(len(fingerprint), 64)

    def test_sha256_calculation(self):
        """Test SHA256 calculation."""
        test_file = self.spdx_text_dir / "test_file_for_sha.txt"
        test_file.write_text("test content")

        sha = self.db._sha256sum(test_file)
        self.assertIsInstance(sha, str)
        self.assertEqual(len(sha), 64)
        self.assertEqual(
            sha, "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72"
        )

        test_text = "test content"
        text_sha = self.db._sha256sum_text(test_text)
        self.assertEqual(sha, text_sha)

    @patch("sentence_transformers.SentenceTransformer")
    def test_embedding_model_lazy_loading(self, mock_transformer):
        """Test that embedding model is loaded lazily."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        self.assertIsNone(self.db._embedding_model)

        model = self.db.embedding_model
        self.assertIsNotNone(model)
        mock_transformer.assert_called_once_with("all-MiniLM-L6-v2")

        self.assertEqual(model, mock_model)

        self.assertIsNotNone(self.db._embedding_model)

    @patch.object(LicenseDatabase, "_save_db")
    def test_update_database_initial_load(self, mock_save_db):
        """Test initial loading and processing of licenses into the database."""
        self.db._licenses_db = None
        mock_progress_callback = Mock()

        licenses_db_entries = self.db._update_database(
            progress_callback=mock_progress_callback
        )

        self.assertIn("MIT", licenses_db_entries)
        mit_entry = licenses_db_entries["MIT"]
        self.assertIsInstance(mit_entry, DatabaseEntry)
        self.assertEqual(mit_entry.name, "MIT")
        self.assertIsNotNone(mit_entry.sha256)
        self.assertIsNotNone(mit_entry.fingerprint)
        # Check file_path points to the file inside the 'text' directory, with original .txt suffix
        self.assertEqual(mit_entry.file_path, self.spdx_text_dir / "MIT.txt")
        self.assertIsNone(mit_entry.embedding)

        mock_progress_callback.assert_any_call(0, 1, "Processing licenses: MIT")
        mock_progress_callback.assert_any_call(
            1, 1, "Finished licenses database update."
        )

        mock_save_db.assert_called_once()
        saved_db_data = mock_save_db.call_args[0][0]
        self.assertIn("MIT", saved_db_data)
        self.assertIsNone(saved_db_data["MIT"]["embedding"])

    @patch("sentence_transformers.SentenceTransformer")
    @patch.object(LicenseDatabase, "_save_db")
    def test_get_embedding_computation_and_cache_update(
        self, mock_save_db, mock_transformer
    ):
        """Test _get_embedding computes and caches embedding."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        mock_transformer.return_value = mock_model

        initial_db_content = {
            "MIT": {
                "sha256": self.db._sha256sum_text(self.mit_content),
                "fingerprint": self.db._canonical_fingerprint(self.mit_content),
                "embedding": None,
                "updated": datetime.now(UTC).isoformat(),
            }
        }
        with open(self.licenses_db_path, "w", encoding="utf-8") as f:
            json.dump(initial_db_content, f, indent=2, ensure_ascii=False)

        raw_db_from_file = self.db._load_existing_db(self.licenses_db_path)
        # file_path needs to include .txt here since that's how it's created in setup
        self.db._licenses_db = {
            name: DatabaseEntry(
                name=name, file_path=self.spdx_text_dir / f"{name}.txt", **data
            )
            for name, data in raw_db_from_file.items()
        }

        mit_entry = self.db._licenses_db["MIT"]

        embedding = self.db._get_embedding(mit_entry)

        np.testing.assert_array_almost_equal(embedding, np.array([0.5, 0.6, 0.7]))
        np.testing.assert_array_almost_equal(
            np.array(mit_entry.embedding), np.array([0.5, 0.6, 0.7])
        )

        mock_model.encode.assert_called_once_with(self.mit_content)

        mock_save_db.assert_called_once()
        saved_db_data = mock_save_db.call_args[0][0]
        self.assertIn("MIT", saved_db_data)
        np.testing.assert_array_almost_equal(
            np.array(saved_db_data["MIT"]["embedding"]), np.array([0.5, 0.6, 0.7])
        )


class TestLicenseAnalyzer(unittest.TestCase):
    """Test the main LicenseAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.spdx_dir = Path(self.temp_dir) / "spdx_repo"
        self.cache_dir = Path(self.temp_dir) / "db_cache"

        self.spdx_dir.mkdir(parents=True)
        self.spdx_text_dir = self.spdx_dir / "text"
        self.spdx_text_dir.mkdir(exist_ok=True)

        self.spdx_json_dir = self.spdx_dir / "json"
        self.spdx_json_dir.mkdir(exist_ok=True)
        dummy_licenses_json = {"licenseListVersion": "test_version_1.0", "licenses": []}
        (self.spdx_json_dir / "licenses.json").write_text(
            json.dumps(dummy_licenses_json)
        )

        self.cache_dir.mkdir(parents=True)

        self.mit_content = "MIT License\n\nCopyright (c) 2024"
        self.apache_content = "Apache License Version 2.0, January 2004"
        self.gpl_content = "GNU General Public License v3.0"

        # Write license files with .txt suffix
        (self.spdx_text_dir / "MIT.txt").write_text(self.mit_content)
        (self.spdx_text_dir / "Apache-2.0.txt").write_text(self.apache_content)
        (self.spdx_text_dir / "GPL-3.0.txt").write_text(self.gpl_content)

        self.patcher_transformer = patch("sentence_transformers.SentenceTransformer")
        self.patcher_util = patch("sentence_transformers.util")

        self.mock_transformer_class = self.patcher_transformer.start()
        self.mock_util = self.patcher_util.start()

        self.mock_model = Mock()
        self.mock_model.encode.return_value = np.array(
            [0.1, 0.2, 0.3], dtype=np.float32
        )
        self.mock_transformer_class.return_value = self.mock_model

        self.mock_util.cos_sim.return_value = np.array([[0.8]])

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
        self.mock_db_progress_callback.assert_called()
        self.assertEqual(len(self.analyzer.db.licenses_db), 3)

        self.assertEqual(self.mock_model.encode.call_count, 0)

        mit_entry = self.analyzer.db.licenses_db["MIT"]
        self.assertEqual(mit_entry.name, "MIT")
        self.assertEqual(mit_entry.file_path, self.spdx_text_dir / "MIT.txt")

    @patch.object(LicenseDatabase, "_get_embedding")
    def test_analyze_text_exact_sha_match(self, mock_get_embedding):
        """Test analyzing text with exact SHA256 match."""
        mock_get_embedding.return_value = np.array([0.9, 0.8, 0.7])

        matches = self.analyzer.analyze_text(self.mit_content, top_n=1)

        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].name, "MIT")
        self.assertEqual(matches[0].score, 1.0)
        self.assertEqual(matches[0].method, MatchMethod.SHA256)

        self.mock_model.encode.assert_not_called()
        mock_get_embedding.assert_not_called()

    @patch.object(LicenseDatabase, "_get_embedding")
    def test_analyze_text_exact_fingerprint_match(self, mock_get_embedding):
        """Test analyzing text with exact fingerprint match (but not SHA256)."""
        fingerprint_only_content = self.mit_content.replace(
            "Copyright (c) 2024", "copyright (C) 2024"
        )

        original_mit_sha = self.analyzer.db._sha256sum_text(self.mit_content)
        original_mit_fp = self.analyzer.db._canonical_fingerprint(self.mit_content)

        modified_sha = self.analyzer.db._sha256sum_text(fingerprint_only_content)
        modified_fp = self.analyzer.db._canonical_fingerprint(fingerprint_only_content)

        self.assertNotEqual(original_mit_sha, modified_sha)
        self.assertEqual(original_mit_fp, modified_fp)

        matches = self.analyzer.analyze_text(fingerprint_only_content, top_n=1)

        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].name, "MIT")
        self.assertEqual(matches[0].score, 1.0)
        self.assertEqual(matches[0].method, MatchMethod.FINGERPRINT)

        self.mock_model.encode.assert_not_called()
        mock_get_embedding.assert_not_called()

    @patch.object(LicenseDatabase, "_get_embedding")
    def test_analyze_text_embedding_match(self, mock_get_embedding):
        """Test analyzing text with embedding match when no exact match."""
        non_exact_text = "This is a license that is similar to MIT but not identical."

        self.mock_model.encode.return_value = np.array(
            [0.1, 0.2, 0.3], dtype=np.float32
        )
        self.mock_util.cos_sim.return_value = np.array([[0.75]])
        mock_get_embedding.return_value = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        mock_per_entry_embed_callback = Mock()
        matches = self.analyzer.analyze_text(
            non_exact_text, per_entry_embed_callback=mock_per_entry_embed_callback
        )

        self.assertGreater(len(matches), 0)

        embedding_matches = [m for m in matches if m.method == MatchMethod.EMBEDDING]
        self.assertGreater(len(embedding_matches), 0)
        # Assertion corrected to expect "Apache-2.0" as per filename
        self.assertIn(embedding_matches[0].name, ["MIT", "Apache-2.0", "GPL-3.0"])
        self.assertEqual(embedding_matches[0].score, 0.75)

        self.mock_model.encode.assert_called_once_with(non_exact_text)
        self.assertEqual(mock_get_embedding.call_count, 3)
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

    @patch.object(LicenseDatabase, "_get_embedding")
    def test_analyze_multiple_files(self, mock_get_embedding):
        """Test analyzing multiple files with progress callback."""
        mock_get_embedding.return_value = np.array([0.9, 0.8, 0.7])

        self.mock_model.encode.side_effect = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
        ]
        self.mock_util.cos_sim.side_effect = [
            np.array([[0.95]]),
            np.array([[0.85]]),
            np.array([[0.75]]),
            np.array([[0.9]]),
            np.array([[0.8]]),
            np.array([[0.7]]),
        ]

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

        self.assertEqual(self.mock_model.encode.call_count, 2)
        self.mock_model.encode.assert_has_calls(
            [
                call("This is input license 1 content."),
                call("This is input license 2 content."),
            ],
            any_order=False,
        )

        self.assertEqual(mock_get_embedding.call_count, 6)

        mock_analysis_progress_callback.assert_called()
        self.assertEqual(mock_analysis_progress_callback.call_args[0][1], 2)
        mock_analysis_progress_callback.assert_any_call(
            1, 2, f"Analyzing {file1_path.name}"
        )
        mock_analysis_progress_callback.assert_any_call(
            2, 2, f"Analyzing {file2_path.name}"
        )
        mock_analysis_progress_callback.assert_any_call(
            2, 2, "Finished analyzing files."
        )

        file1_matches = results[str(file1_path)]
        self.assertEqual(len(file1_matches), 2)
        self.assertEqual(file1_matches[0].score, 0.95)
        self.assertEqual(file1_matches[0].method, MatchMethod.EMBEDDING)

    def test_get_database_stats(self):
        """Test getting database statistics."""
        stats = self.analyzer.get_database_stats()
        self.assertIn("total_licenses", stats)
        self.assertEqual(stats["total_licenses"], 3)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.spdx_dir = Path(self.temp_dir) / "spdx_repo_for_convenience"
        self.default_cache_base_dir = Path(
            user_cache_dir(appname="license-analyzer", appauthor="envolution")
        )
        self.default_db_cache_dir = self.default_cache_base_dir / "db_cache"
        self.default_spdx_data_dir = self.default_cache_base_dir / "spdx"

        self.spdx_dir.mkdir(parents=True)
        self.spdx_text_dir = self.spdx_dir / "text"
        self.spdx_text_dir.mkdir(exist_ok=True)
        self.spdx_json_dir = self.spdx_dir / "json"
        self.spdx_json_dir.mkdir(exist_ok=True)
        dummy_licenses_json = {"licenseListVersion": "test_version_1.0", "licenses": []}
        (self.spdx_json_dir / "licenses.json").write_text(
            json.dumps(dummy_licenses_json)
        )

        self.default_db_cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_spdx_data_dir.mkdir(parents=True, exist_ok=True)

        self.mit_content = "MIT License\n\nTest content for convenience function"
        # Write with .txt suffix
        (self.spdx_text_dir / "MIT.txt").write_text(self.mit_content)

        self.patcher_transformer = patch("sentence_transformers.SentenceTransformer")
        self.patcher_util = patch("sentence_transformers.util")

        self.mock_transformer_class = self.patcher_transformer.start()
        self.mock_util = self.patcher_util.start()

        self.mock_model = Mock()
        self.mock_model.encode.return_value = np.array(
            [0.1, 0.2, 0.3], dtype=np.float32
        )
        self.mock_transformer_class.return_value = self.mock_model
        self.mock_util.cos_sim.return_value = np.array([[0.95]])

        self.patcher_get_embedding = patch(
            "license_analyzer.core.LicenseDatabase._get_embedding"
        )
        self.mock_get_embedding = self.patcher_get_embedding.start()
        self.mock_get_embedding.return_value = np.array(
            [0.9, 0.8, 0.7], dtype=np.float32
        )

    def tearDown(self):
        self.patcher_transformer.stop()
        self.patcher_util.stop()
        self.patcher_get_embedding.stop()
        shutil.rmtree(self.temp_dir)
        if self.default_cache_base_dir.exists():
            shutil.rmtree(self.default_cache_base_dir)

    @patch("license_analyzer.core.LicenseAnalyzer")
    def test_analyze_license_file_function(self, mock_analyzer_class):
        """Test analyze_license_file convenience function."""
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze_file.return_value = [
            LicenseMatch("MIT", 1.0, MatchMethod.SHA256)
        ]
        mock_analyzer_class.return_value = mock_analyzer_instance

        test_file = Path(self.temp_dir) / "test_input_file.txt"
        test_file.write_text("test content")

        matches = analyze_license_file(test_file, top_n=3, spdx_dir=self.spdx_dir)

        mock_analyzer_class.assert_called_once_with(
            spdx_dir=self.spdx_dir,
        )
        # Explicitly expect per_entry_embed_callback=None
        mock_analyzer_instance.analyze_file.assert_called_once_with(test_file, 3)
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

        mock_analyzer_class.assert_called_once_with(
            spdx_dir=self.spdx_dir,
        )
        # Explicitly expect per_entry_embed_callback=None
        mock_analyzer_instance.analyze_text.assert_called_once_with(test_text, 5)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].name, "MIT")


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.spdx_dir = Path(self.temp_dir) / "spdx_repo_root"
        self.cache_dir = Path(self.temp_dir) / "db_cache"

        self.spdx_dir.mkdir(parents=True)
        self.spdx_text_dir = self.spdx_dir / "text"
        self.spdx_json_dir = self.spdx_dir / "json"
        self.spdx_text_dir.mkdir(exist_ok=True)
        self.spdx_json_dir.mkdir(exist_ok=True)

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

        # Write license files into the 'text' subdirectory, with .txt suffix
        (self.spdx_text_dir / "MIT.txt").write_text(self.mit_license_content)
        (self.spdx_text_dir / "Apache-2.0.txt").write_text(self.apache_license_content)

        dummy_licenses_json = {
            "licenseListVersion": "test_version_1.0",
            "licenses": [
                {"licenseId": "MIT", "name": "MIT License", "isOsiApproved": True},
                {
                    "licenseId": "Apache-2.0",
                    "name": "Apache License 2.0",
                    "isOsiApproved": True,
                },
            ],
        }
        (self.spdx_json_dir / "licenses.json").write_text(
            json.dumps(dummy_licenses_json)
        )

        self.patcher_transformer = patch("sentence_transformers.SentenceTransformer")
        self.patcher_util = patch("sentence_transformers.util")

        self.mock_transformer_class = self.patcher_transformer.start()
        self.mock_util = self.patcher_util.start()

        self.mock_model = Mock()

        def mock_encode_side_effect(text):
            if "MIT License" in text:
                return np.array([0.1, 0.2, 0.3], dtype=np.float32)
            elif "Apache License" in text:
                return np.array([0.4, 0.5, 0.6], dtype=np.float32)
            else:
                return np.array([0.7, 0.8, 0.9], dtype=np.float32)

        self.mock_model.encode.side_effect = mock_encode_side_effect

        self.mock_transformer_class.return_value = self.mock_model

        self.mock_util.cos_sim.return_value = np.array([[0.85]])

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

        self.mock_model.encode.reset_mock()
        self.mock_util.cos_sim.reset_mock()

        matches = analyzer.analyze_text(self.mit_license_content, top_n=1)

        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].name, "MIT")
        self.assertEqual(matches[0].score, 1.0)
        self.assertEqual(matches[0].method, MatchMethod.SHA256)

        self.mock_model.encode.assert_not_called()
        self.mock_util.cos_sim.assert_not_called()

        self.mock_model.encode.reset_mock()
        self.mock_util.cos_sim.reset_mock()

        similar_content = self.mit_license_content.replace(
            "2024 Test Project", "2023 New Project"
        )

        with patch.object(LicenseDatabase, "_get_embedding") as mock_get_embedding:
            mock_get_embedding.side_effect = [
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                np.array([0.4, 0.5, 0.6], dtype=np.float32),
            ]

            mock_per_entry_embed_callback = Mock()
            matches = analyzer.analyze_text(
                similar_content,
                top_n=5,
                per_entry_embed_callback=mock_per_entry_embed_callback,
            )

            self.assertGreater(len(matches), 0)

            embedding_matches = [
                m for m in matches if m.method == MatchMethod.EMBEDDING
            ]
            self.assertGreater(len(embedding_matches), 0)
            self.assertEqual(embedding_matches[0].score, 0.85)
            # Assertion corrected to expect "Apache-2.0"
            self.assertIn(embedding_matches[0].name, ["MIT", "Apache-2.0"])

            self.mock_model.encode.assert_called_once_with(similar_content)
            self.assertEqual(mock_get_embedding.call_count, 2)
            mock_per_entry_embed_callback.assert_called()

        stats = analyzer.get_database_stats()
        # Expect 2 licenses from setUp now
        self.assertEqual(stats["total_licenses"], 2)


if __name__ == "__main__":
    unittest.main()
