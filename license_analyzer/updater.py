# license_analyzer/updater.py
import requests
import json
import shutil
import tarfile
import re
import logging
from pathlib import Path
from datetime import date, datetime, UTC
from typing import Optional, Dict, Tuple
from appdirs import user_cache_dir

logger = logging.getLogger(__name__)

SPDX_GITHUB_REPO = "spdx/license-list-data"
SPDX_GITHUB_API_RELEASES = f"https://api.github.com/repos/{SPDX_GITHUB_REPO}/releases/latest"

APP_NAME = "license-analyzer"
APP_AUTHOR = "envolution"

class LicenseUpdater:
    """
    Manages downloading and updating SPDX license data from GitHub.
    """
    def __init__(self, cache_dir: Optional[Path] = None, spdx_data_dir: Optional[Path] = None):
        if cache_dir is None:
            # Use appdirs for cross-platform cache directory
            self.cache_base_dir = Path(user_cache_dir(appname=APP_NAME, appauthor=APP_AUTHOR))
        else:
            self.cache_base_dir = Path(cache_dir)

        # Directory where updater stores its internal state (last checked date, version)
        self.updater_cache_dir = self.cache_base_dir / "updater"
        self.updater_cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_update_info_path = self.updater_cache_dir / "last_update_info.json"

        # Directory where the actual SPDX license files will be stored, used by LicenseDatabase
        if spdx_data_dir is None:
            self.spdx_data_dir = self.cache_base_dir / "spdx"
        else:
            self.spdx_data_dir = Path(spdx_data_dir)

        self.spdx_data_dir.mkdir(parents=True, exist_ok=True)

    def _get_last_checked_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Reads the last checked date and version from cache."""
        if not self.last_update_info_path.exists():
            return None, None
        try:
            with open(self.last_update_info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            return info.get("last_version"), info.get("last_checked_date")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read last update info from {self.last_update_info_path}: {e}")
            return None, None

    def _set_last_checked_info(self, version: str, checked_date: str) -> None:
        """Writes the current version and checked date to cache."""
        info = {"last_version": version, "last_checked_date": checked_date}
        try:
            with open(self.last_update_info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Failed to write last update info to {self.last_update_info_path}: {e}")

    def _fetch_latest_release_info(self) -> Optional[Dict]:
        """Fetches latest release information from GitHub API."""
        try:
            response = requests.get(SPDX_GITHUB_API_RELEASES, timeout=10)
            response.raise_for_status() # Raise an exception for HTTP errors
            release_info = response.json()
            return release_info
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch latest release info from GitHub: {e}")
            return None

    def _parse_version_from_tag(self, tag_name: str) -> Optional[Tuple[int, ...]]:
        """Parses a numerical version tuple from a tag name like 'v3.24' or '3.24.0'."""
        # Remove 'v' prefix if present
        version_str = tag_name.lstrip('v')
        # Split by dot and convert to integers
        try:
            return tuple(map(int, version_str.split('.')))
        except ValueError:
            logger.warning(f"Could not parse version from tag: {tag_name}")
            return None

    def _download_and_extract_licenses(self, tarball_url: str, version: str) -> bool:
        """
        Downloads the tarball and extracts license texts to spdx_data_dir.
        """
        download_path = self.updater_cache_dir / f"{version}.tar.gz"
        logger.info(f"Downloading license data from {tarball_url} to {download_path}")

        try:
            with requests.get(tarball_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(download_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info("Download complete.")

            # Clear existing data in spdx_data_dir before extracting new ones
            if self.spdx_data_dir.exists():
                logger.info(f"Clearing existing license data in {self.spdx_data_dir}")
                # Keep the root directory, just clear its contents
                for item in self.spdx_data_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            else:
                self.spdx_data_dir.mkdir(parents=True, exist_ok=True)


            logger.info(f"Extracting license data to {self.spdx_data_dir}")
            with tarfile.open(download_path, 'r:gz') as tar:
                # Find the root directory inside the tar (e.g., license-list-data-3.24)
                # It's usually the first entry or the one with a single component path
                root_dir_name = None
                for member in tar.getmembers():
                    # Check if it's a directory at the root level (e.g. license-list-data-3.24/)
                    if '/' not in member.name and member.isdir():
                        root_dir_name = member.name
                        break
                if root_dir_name is None:
                    raise ValueError("Could not find root directory in tarball.")

                # Extract only 'text' and 'text/exceptions' directories
                # We need to manually iterate and extract to target self.spdx_data_dir directly
                for member in tar.getmembers():
                    if member.name.startswith(f"{root_dir_name}/text/"):
                        # Calculate the target path relative to spdx_data_dir
                        relative_path = Path(member.name).relative_to(f"{root_dir_name}/text/")
                        target_path = self.spdx_data_dir / relative_path

                        if member.isdir():
                            target_path.mkdir(parents=True, exist_ok=True)
                        elif member.isfile():
                            # Extract file
                            with tar.extractfile(member) as source_file, open(target_path, 'wb') as dest_file:
                                shutil.copyfileobj(source_file, dest_file)
            logger.info("Extraction complete.")
            return True
        except (requests.exceptions.RequestException, tarfile.TarError, IOError, ValueError) as e:
            logger.error(f"Failed to download or extract license data: {e}")
            return False
        finally:
            if download_path.exists():
                download_path.unlink() # Clean up the tarball

    def check_for_updates(self, force: bool = False) -> bool:
        """
        Checks for and applies updates to the SPDX license database.

        Args:
            force: If True, forces an update check regardless of last check date.

        Returns:
            True if an update was performed, False otherwise.
        """
        last_version, last_checked_date_str = self._get_last_checked_info()
        today_str = date.today().isoformat()

        if not force and last_checked_date_str == today_str and self.spdx_data_dir.exists() and any(self.spdx_data_dir.iterdir()):
            logger.info(f"License data already checked today ({today_str}) and data exists. Skipping update check.")
            return False # No update performed

        logger.info("Checking for new SPDX license data releases...")
        release_info = self._fetch_latest_release_info()
        if not release_info:
            logger.error("Could not get latest release info. Cannot check for updates.")
            return False

        latest_tag = release_info.get("tag_name")
        tarball_url = release_info.get("tarball_url")

        if not latest_tag or not tarball_url:
            logger.error("Latest release info missing tag_name or tarball_url.")
            return False

        latest_version_parsed = self._parse_version_from_tag(latest_tag)
        last_version_parsed = self._parse_version_from_tag(last_version) if last_version else None

        # Check if SPDX data directory is empty or if it needs initial population
        spdx_data_exists = self.spdx_data_dir.exists() and any(self.spdx_data_dir.iterdir())
        if not spdx_data_exists:
            logger.info("SPDX data directory is empty or missing. Performing initial download.")
            if self._download_and_extract_licenses(tarball_url, latest_tag):
                self._set_last_checked_info(latest_tag, today_str)
                logger.info(f"Successfully initialized SPDX license data to version {latest_tag}.")
                return True
            else:
                logger.error("Initial download of SPDX license data failed.")
                return False


        if last_version_parsed and latest_version_parsed and latest_version_parsed <= last_version_parsed and not force:
            logger.info(f"Current license data version '{last_version}' is up-to-date with latest '{latest_tag}'.")
            self._set_last_checked_info(last_version, today_str) # Still mark as checked today
            return False

        logger.info(f"New SPDX license data available: {latest_tag}. Updating from {last_version or 'N/A'}...")
        if self._download_and_extract_licenses(tarball_url, latest_tag):
            self._set_last_checked_info(latest_tag, today_str)
            logger.info(f"Successfully updated SPDX license data to version {latest_tag}.")
            return True
        else:
            logger.error("Failed to update SPDX license data.")
            return False

    def get_spdx_data_path(self) -> Path:
        """Returns the path where SPDX license data is stored."""
        return self.spdx_data_dir
