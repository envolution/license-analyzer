# license_analyzer/__init__.py
"""
The license_analyzer package.
"""

# Expose core components for easy import
from .core import LicenseAnalyzer, LicenseMatch, MatchMethod
from .core import analyze_license_file, analyze_license_text  # Convenience functions
from .updater import LicenseUpdater  # NEW import

# Define package version
__version__ = "0.0.3"  # Keep in sync with pyproject.toml
