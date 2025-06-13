# tests/test_license_analyzer.py
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import the module components
from license_analyzer import (
    LicenseAnalyzer, 
    LicenseDatabase, 
    LicenseMatch, 
    DatabaseEntry,
    MatchMethod,
    analyze_license_file,
    analyze_license_text
)


class TestLicenseMatch(unittest.TestCase):
    """Test the LicenseMatch dataclass."""
    
    def test_valid_match(self):
        """Test creating a valid LicenseMatch."""
        match = LicenseMatch(
            name="MIT.txt",
            score=0.95,
            method=MatchMethod.EMBEDDING,
            license_type="license"
        )
        self.assertEqual(match.name, "MIT.txt")
        self.assertEqual(match.score, 0.95)
        self.assertEqual(match.method, MatchMethod.EMBEDDING)
        self.assertEqual(match.license_type, "license")
    
    def test_invalid_score_low(self):
        """Test that scores below 0.0 raise ValueError."""
        with self.assertRaises(ValueError):
            LicenseMatch(
                name="test.txt",
                score=-0.1,
                method=MatchMethod.SHA256
            )
    
    def test_invalid_score_high(self):
        """Test that scores above 1.0 raise ValueError."""
        with self.assertRaises(ValueError):
            LicenseMatch(
                name="test.txt",
                score=1.1,
                method=MatchMethod.SHA256
            )


class TestDatabaseEntry(unittest.TestCase):
    """Test the DatabaseEntry dataclass."""
    
    def test_database_entry_creation(self):
        """Test creating a DatabaseEntry."""
        entry = DatabaseEntry(
            name="MIT.txt",
            sha256="abcd1234",
            fingerprint="efgh5678",
            embedding=[0.1, 0.2, 0.3],
            file_path=Path("/test/MIT.txt")
        )
        self.assertEqual(entry.name, "MIT.txt")
        self.assertEqual(entry.sha256, "abcd1234")
        self.assertEqual(entry.fingerprint, "efgh5678")
        self.assertEqual(entry.embedding, [0.1, 0.2, 0.3])
        self.assertEqual(entry.file_path, Path("/test/MIT.txt"))


class TestLicenseDatabase(unittest.TestCase):
    """Test the LicenseDatabase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.spdx_dir = Path(self.temp_dir) / "spdx"
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.exceptions_dir = self.spdx_dir / "exceptions"
        
        # Create directory structure
        self.spdx_dir.mkdir(parents=True)
        self.exceptions_dir.mkdir(parents=True)
        self.cache_dir.mkdir(parents=True)
        
        # Create test license files
        mit_content = """MIT License

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
        
        (self.spdx_dir / "MIT.txt").write_text(mit_content)
        (self.exceptions_dir / "TestException.txt").write_text("Test exception content")
        
        self.db = LicenseDatabase(self.spdx_dir, self.cache_dir, "all-MiniLM-L6-v2")
    
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
        test_file = self.spdx_dir / "test.txt"
        test_file.write_text("test content")
        
        sha = self.db._sha256sum(test_file)
        self.assertIsInstance(sha, str)
        self.assertEqual(len(sha), 64)
        
        # Test text SHA256
        text_sha = self.db._sha256sum_text("test content")
        self.assertEqual(sha, text_sha)
    
    @patch('license_analyzer.SentenceTransformer')
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
    
    def test_database_loading(self):
        """Test database loading and updating."""
        # Access databases to trigger loading
        licenses_db = self.db.licenses_db
        exceptions_db = self.db.exceptions_db
        
        self.assertIn("MIT.txt", licenses_db)
        self.assertIn("TestException.txt", exceptions_db)
        
        # Check database entry structure
        mit_entry = licenses_db["MIT.txt"]
        self.assertIsInstance(mit_entry, DatabaseEntry)
        self.assertEqual(mit_entry.name, "MIT.txt")
        self.assertIsNotNone(mit_entry.sha256)
        self.assertIsNotNone(mit_entry.fingerprint)


class TestLicenseAnalyzer(unittest.TestCase):
    """Test the main LicenseAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.spdx_dir = Path(self.temp_dir) / "spdx"
        self.cache_dir = Path(self.temp_dir) / "cache"
        
        # Create directory structure
        self.spdx_dir.mkdir(parents=True)
        self.cache_dir.mkdir(parents=True)
        
        # Create test license file
        mit_content = "MIT License\n\nCopyright (c) 2024"
        (self.spdx_dir / "MIT.txt").write_text(mit_content)
        
        # Mock the sentence transformer to avoid downloading models in tests
        with patch('license_analyzer.SentenceTransformer') as mock_transformer:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            mock_transformer.return_value = mock_model
            
            self.analyzer = LicenseAnalyzer(
                spdx_dir=self.spdx_dir,
                cache_dir=self.cache_dir
            )
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, LicenseAnalyzer)
        self.assertIsInstance(self.analyzer.db, LicenseDatabase)
    
    @patch('license_analyzer.util')
    @patch('license_analyzer.SentenceTransformer')
    def test_analyze_text_exact_match(self, mock_transformer, mock_util):
        """Test analyzing text with exact SHA256 match."""
        # Setup mocks
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        # Test exact match
        mit_content = "MIT License\n\nCopyright (c) 2024"
        matches = self.analyzer.analyze_text(mit_content)
        
        self.assertGreater(len(matches), 0)
        # Should have at least one perfect match
        perfect_matches = [m for m in matches if m.score == 1.0]
        self.assertGreater(len(perfect_matches), 0)
    
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
    
    @patch('license_analyzer.util')
    @patch('license_analyzer.SentenceTransformer')
    def test_analyze_multiple_files(self, mock_transformer, mock_util):
        """Test analyzing multiple files."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_transformer.return_value = mock_model
        mock_util.cos_sim.return_value = [[0.8]]
        
        # Create test files
        file1 = Path(self.temp_dir) / "license1.txt"
        file2 = Path(self.temp_dir) / "license2.txt"
        file1.write_text("MIT License content")
        file2.write_text("Apache License content")
        
        results = self.analyzer.analyze_multiple_files([file1, file2])
        
        self.assertEqual(len(results), 2)
        self.assertIn(str(file1), results)
        self.assertIn(str(file2), results)
    
    def test_get_database_stats(self):
        """Test getting database statistics."""
        stats = self.analyzer.get_database_stats()
        
        self.assertIn("licenses", stats)
        self.assertIn("exceptions", stats)
        self.assertIn("total", stats)
        self.assertIsInstance(stats["licenses"], int)
        self.assertIsInstance(stats["exceptions"], int)
        self.assertEqual(stats["total"], stats["licenses"] + stats["exceptions"])


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.spdx_dir = Path(self.temp_dir) / "spdx"
        self.spdx_dir.mkdir(parents=True)
        
        # Create test license
        (self.spdx_dir / "MIT.txt").write_text("MIT License\n\nTest content")
    
    @patch('license_analyzer.LicenseAnalyzer')
    def test_analyze_license_file_function(self, mock_analyzer_class):
        """Test analyze_license_file convenience function."""
        mock_analyzer = Mock()
        mock_analyzer.analyze_file.return_value = [
            LicenseMatch("MIT.txt", 1.0, MatchMethod.SHA256)
        ]
        mock_analyzer_class.return_value = mock_analyzer
        
        test_file = self.spdx_dir / "test.txt"
        test_file.write_text("test content")
        
        matches = analyze_license_file(test_file, top_n=3, spdx_dir=self.spdx_dir)
        
        mock_analyzer_class.assert_called_once_with(spdx_dir=self.spdx_dir)
        mock_analyzer.analyze_file.assert_called_once_with(test_file, 3)
        self.assertEqual(len(matches), 1)
    
    @patch('license_analyzer.LicenseAnalyzer')
    def test_analyze_license_text_function(self, mock_analyzer_class):
        """Test analyze_license_text convenience function."""
        mock_analyzer = Mock()
        mock_analyzer.analyze_text.return_value = [
            LicenseMatch("MIT.txt", 0.95, MatchMethod.EMBEDDING)
        ]
        mock_analyzer_class.return_value = mock_analyzer
        
        test_text = "MIT License test"
        matches = analyze_license_text(test_text, top_n=5, spdx_dir=self.spdx_dir)
        
        mock_analyzer_class.assert_called_once_with(spdx_dir=self.spdx_dir)
        mock_analyzer.analyze_text.assert_called_once_with(test_text, 5)
        self.assertEqual(len(matches), 1)


class TestMatchMethodEnum(unittest.TestCase):
    """Test the MatchMethod enum."""
    
    def test_match_method_values(self):
        """Test MatchMethod enum values."""
        self.assertEqual(MatchMethod.SHA256.value, "sha256")
        self.assertEqual(MatchMethod.FINGERPRINT.value, "fingerprint")
        self.assertEqual(MatchMethod.EMBEDDING.value, "embedding")


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.spdx_dir = Path(self.temp_dir) / "spdx"
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.exceptions_dir = self.spdx_dir / "exceptions"
        
        # Create directory structure
        self.spdx_dir.mkdir(parents=True)
        self.exceptions_dir.mkdir(parents=True)
        
        # Create realistic license content
        mit_license = """MIT License

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
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
        
        apache_license = """Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions."""
        
        (self.spdx_dir / "MIT.txt").write_text(mit_license)
        (self.spdx_dir / "Apache-2.0.txt").write_text(apache_license)
        (self.exceptions_dir / "TestException.txt").write_text("Test exception content")
    
    @patch('license_analyzer.util')
    @patch('license_analyzer.SentenceTransformer')
    def test_full_workflow(self, mock_transformer, mock_util):
        """Test complete workflow from initialization to analysis."""
        # Setup mocks
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3] * 128)  # Realistic embedding size
        mock_transformer.return_value = mock_model
        mock_util.cos_sim.return_value = [[0.85]]
        
        # Initialize analyzer
        analyzer = LicenseAnalyzer(spdx_dir=self.spdx_dir, cache_dir=self.cache_dir)
        
        # Test exact match
        mit_content = (self.spdx_dir / "MIT.txt").read_text()
        matches = analyzer.analyze_text(mit_content, top_n=3)
        
        self.assertGreater(len(matches), 0)
        # Should find exact match
        exact_matches = [m for m in matches if m.score == 1.0]
        self.assertGreater(len(exact_matches), 0)
        
        # Test similarity match
        similar_content = "MIT License\n\nCopyright (c) 2024 Different Project\n\nPermission is hereby granted..."
        matches = analyzer.analyze_text(similar_content, top_n=5)
        
        self.assertGreater(len(matches), 0)
        
        # Test database stats
        stats = analyzer.get_database_stats()
        self.assertEqual(stats["licenses"], 2)  # MIT and Apache
        self.assertEqual(stats["exceptions"], 1)  # TestException
        self.assertEqual(stats["total"], 3)


if __name__ == '__main__':
    unittest.main()


# tests/test_cli.py
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from io import StringIO
import sys

# Import CLI module (assuming it's in the same package)
from license_analyzer.cli import (
    format_text_output,
    format_json_output,
    format_csv_output,
    main
)
from license_analyzer import LicenseMatch, MatchMethod


class TestCLIFormatting(unittest.TestCase):
    """Test CLI output formatting functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.matches = [
            LicenseMatch("MIT.txt", 1.0, MatchMethod.SHA256, "license"),
            LicenseMatch("Apache-2.0.txt", 0.95, MatchMethod.EMBEDDING, "license"),
            LicenseMatch("GPL-3.0.txt", 0.85, MatchMethod.FINGERPRINT, "license")
        ]
    
    def test_format_text_output(self):
        """Test text output formatting."""
        output = format_text_output("test.txt", self.matches)
        
        self.assertIn("Analysis results for: test.txt", output)
        self.assertIn("MIT.txt", output)
        self.assertIn("1.0000", output)
        self.assertIn("sha256", output)
        self.assertIn("Apache-2.0.txt", output)
        self.assertIn("0.9500", output)
    
    def test_format_json_output(self):
        """Test JSON output formatting."""
        results = {"test.txt": self.matches}
        output = format_json_output(results)
        
        parsed = json.loads(output)
        self.assertIn("test.txt", parsed)
        self.assertEqual(len(parsed["test.txt"]), 3)
        self.assertEqual(parsed["test.txt"][0]["name"], "MIT.txt")
        self.assertEqual(parsed["test.txt"][0]["score"], 1.0)
        self.assertEqual(parsed["test.txt"][0]["method"], "sha256")
    
    def test_format_csv_output(self):
        """Test CSV output formatting."""
        results = {"test.txt": self.matches}
        output = format_csv_output(results)
        
        lines = output.split('\n')
        self.assertEqual(lines[0], "file_path,license_name,score,method,license_type")
        self.assertIn('"test.txt","MIT.txt",1.0,sha256,license', lines[1])
        self.assertIn('"test.txt","Apache-2.0.txt",0.95,embedding,license', lines[2])
    
    def test_format_empty_matches(self):
        """Test formatting with no matches."""
        output = format_text_output("empty.txt", [])
        self.assertIn("No matches found", output)
        
        json_output = format_json_output({"empty.txt": []})
        parsed = json.loads(json_output)
        self.assertEqual(parsed["empty.txt"], [])


class TestCLIMain(unittest.TestCase):
    """Test CLI main function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_license.txt"
        self.test_file.write_text("MIT License\n\nTest content")
    
    @patch('license_analyzer.cli.LicenseAnalyzer')
    @patch('sys.argv', ['cli.py', 'test.txt'])
    def test_main_single_file(self, mock_analyzer_class):
        """Test main function with single file."""
        mock_analyzer = Mock()
        mock_analyzer.analyze_file.return_value = [
            LicenseMatch("MIT.txt", 1.0, MatchMethod.SHA256)
        ]
        mock_analyzer.get_database_stats.return_value = {
            "licenses": 10, "exceptions": 5, "total": 15
        }
        mock_analyzer_class.return_value = mock_analyzer
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with patch('pathlib.Path.exists', return_value=True):
                try:
                    main()
                    output = mock_stdout.getvalue()
                    self.assertIn("MIT.txt", output)
                    self.assertIn("1.0000", output)
                except SystemExit:
                    pass  # Expected for argument parsing
    
    @patch('sys.argv', ['cli.py'])
    def test_main_no_args(self):
        """Test main function with no arguments."""
        with patch('sys.stderr', new_callable=StringIO):
            with self.assertRaises(SystemExit):
                main()


if __name__ == '__main__':
    unittest.main()
