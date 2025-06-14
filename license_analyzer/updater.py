# license_analyzer/updater.py
import requests
import json
import shutil
import tarfile
import re
import logging
from pathlib import Path
from datetime import date, datetime, UTC
from typing import Optional, Dict, Tuple, Callable, Any
from appdirs import user_cache_dir

logger = logging.getLogger(__name__)

SPDX_GITHUB_REPO = "spdx/license-list-data"
SPDX_GITHUB_API_RELEASES = (
    f"https://api.github.com/repos/{SPDX_GITHUB_REPO}/releases/latest"
)

APP_NAME = "license-analyzer"
APP_AUTHOR = "envolution"


class LicenseUpdater:
    """
    Manages downloading and updating SPDX license data from GitHub.
    """

    def __init__(
        self, cache_dir: Optional[Path] = None, spdx_data_dir: Optional[Path] = None
    ):
        if cache_dir is None:
            # Use appdirs for cross-platform cache directory
            self.cache_base_dir = Path(
                user_cache_dir(appname=APP_NAME, appauthor=APP_AUTHOR)
            )
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
            with open(self.last_update_info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            return info.get("last_version"), info.get("last_checked_date")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(
                f"Failed to read last update info from {self.last_update_info_path}: {e}"
            )
            return None, None

    def _set_last_checked_info(self, version: str, checked_date: str) -> None:
        """Writes the current version and checked date to cache."""
        info = {"last_version": version, "last_checked_date": checked_date}
        try:
            with open(self.last_update_info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(
                f"Failed to write last update info to {self.last_update_info_path}: {e}"
            )

    def _fetch_latest_release_info(self) -> Optional[Dict]:
        """Fetches latest release information from GitHub API."""
        try:
            response = requests.get(SPDX_GITHUB_API_RELEASES, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            release_info = response.json()
            return release_info
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch latest release info from GitHub: {e}")
            return None

    def _parse_version_from_tag(self, tag_name: str) -> Optional[Tuple[int, ...]]:
        """Parses a numerical version tuple from a tag name like 'v3.24' or '3.24.0'."""
        # Remove 'v' prefix if present
        version_str = tag_name.lstrip("v")
        # Split by dot and convert to integers
        try:
            return tuple(map(int, version_str.split(".")))
        except ValueError:
            logger.warning(f"Could not parse version from tag: {tag_name}")
            return None

    def _download_and_extract_licenses(
        self,
        tarball_url: str,
        version: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> bool:
        """
        Downloads the tarball and extracts license texts to spdx_data_dir.
        progress_callback: A function (current_bytes, total_bytes, status_message) for UI updates.
        """
        download_path = self.updater_cache_dir / f"{version}.tar.gz"
        logger.debug(f"Downloading license data from {tarball_url} to {download_path}")

        try:
            with requests.get(tarball_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))

                # Initialize progress for download phase
                if progress_callback:
                    # Pass the known total size here, use plain text status message
                    progress_callback(0, total_size, "Downloading SPDX data...")

                downloaded_size = 0
                with open(download_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if progress_callback:
                            # Update completed bytes. The total size is already set.
                            # Use plain text status message
                            progress_callback(
                                downloaded_size,
                                total_size,
                                "Downloading SPDX data...",
                            )
            logger.debug("Download complete.")

            # Clear existing data...
            if self.spdx_data_dir.exists():
                logger.debug(f"Clearing existing license data in {self.spdx_data_dir}")
                for item in self.spdx_data_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            else:
                self.spdx_data_dir.mkdir(parents=True, exist_ok=True)

            logger.debug(f"Extracting license data to {self.spdx_data_dir}")

            # --- BEGIN Extraction Progress Setup ---
            # Re-open tarball to get member list for total count, without extracting yet
            temp_tar_file = tarfile.open(download_path, "r:gz")
            try:
                root_dir_name = None
                for member in temp_tar_file.getmembers():
                    if "/" not in member.name and member.isdir():
                        root_dir_name = member.name
                        break
                if root_dir_name is None:
                    raise ValueError(
                        "Could not find root directory in tarball for counting."
                    )

                # Count only files within 'text/' for accurate progress
                members_to_extract_count = len(
                    [
                        member
                        for member in temp_tar_file.getmembers()
                        if member.name.startswith(f"{root_dir_name}/text/")
                        and member.isfile()
                    ]
                )
            finally:
                temp_tar_file.close()  # Close after counting

            if progress_callback:
                # Reset progress for extraction phase, using file count as total
                # Use plain text status message
                progress_callback(
                    0, members_to_extract_count, "Extracting files..."
                )
            # --- END Extraction Progress Setup ---

            with tarfile.open(download_path, "r:gz") as tar:
                current_root_dir_name = None
                for member in tar.getmembers():
                    if "/" not in member.name and member.isdir():
                        current_root_dir_name = member.name
                        break
                if current_root_dir_name is None:
                    raise ValueError(
                        "Could not find root directory in tarball during extraction."
                    )

                extracted_count = 0
                for member in tar.getmembers():
                    if member.name.startswith(f"{current_root_dir_name}/text/"):
                        relative_path = Path(member.name).relative_to(
                            f"{current_root_dir_name}/text/"
                        )
                        # Store extracted files without .txt suffix, matching database keys
                        target_filename = relative_path.stem if relative_path.suffix == ".txt" else relative_path.name
                        target_path = self.spdx_data_dir / target_filename

                        if member.isdir():
                            target_path.mkdir(parents=True, exist_ok=True)
                        elif member.isfile():  # Only count files for progress
                            with tar.extractfile(member) as source_file, open(
                                target_path, "wb"
                            ) as dest_file:
                                shutil.copyfileobj(source_file, dest_file)
                            extracted_count += 1
                            if progress_callback:
                                # Update progress with files extracted vs total files for extraction
                                # Use plain text status message
                                progress_callback(
                                    extracted_count,
                                    members_to_extract_count,
                                    "Extracting files...",
                                )

            logger.debug("Extraction complete.")
            return True
        except (
            requests.exceptions.RequestException,
            tarfile.TarError,
            IOError,
            ValueError,
        ) as e:
            logger.error(f"Failed to download or extract license data: {e}")
            return False
        finally:
            if download_path.exists():
                download_path.unlink()  # Clean up the tarball

    def check_for_updates(
        self,
        force: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[bool, str]:
        """
        Checks for and applies updates to the SPDX license database.

        Args:
            force: If True, forces an update check regardless of last check date.
            progress_callback: Optional callable for download/extraction progress.

        Returns:
            Tuple[bool, str]: (True if an update was performed, status message).
        """
        last_version, last_checked_date_str = self._get_last_checked_info()
        today_str = date.today().isoformat()

        spdx_data_exists = self.spdx_data_dir.exists() and any(
            self.spdx_data_dir.iterdir()
        )

        # 1. Early exit if already checked today and data exists, unless force update
        if not force and spdx_data_exists and last_checked_date_str == today_str:
            logger.info(f"SPDX data already checked today ({last_version}). Skipping update check.")
            if progress_callback:
                progress_callback(1, 1, f"SPDX data up-to-date ({last_version}).") # Force completion of indeterminate task
            return False, f"License data is already up-to-date ({last_version})."

        # If we reach here, we need to check for updates (either forced, not checked today, or no data yet)
        logger.info("Checking for new SPDX license data releases...")
        
        # Initial status for checking updates, total=0 for indeterminate progress initially
        if progress_callback:
            progress_callback(0, 0, "Checking for updates...") # Indeterminate status

        release_info = self._fetch_latest_release_info()
        if not release_info:
            return False, "Could not get latest release info. Check network connection."

        latest_tag = release_info.get("tag_name")
        tarball_url = release_info.get("tarball_url")

        if not latest_tag or not tarball_url:
            return False, "Latest release info missing tag_name or tarball_url."

        latest_version_parsed = self._parse_version_from_tag(latest_tag)
        # Default to a very old version if no last_version found, ensures initial download
        last_version_parsed = (
            self._parse_version_from_tag(last_version) if last_version else (0,)
        )

        if not spdx_data_exists:
            logger.info(
                "SPDX data directory is empty or missing. Performing initial download."
            )
            if self._download_and_extract_licenses(
                tarball_url, latest_tag, progress_callback
            ):
                self._set_last_checked_info(latest_tag, today_str)
                return (
                    True,
                    f"Successfully initialized SPDX license data to version {latest_tag}.",
                )
            else:
                return False, "Initial download of SPDX license data failed."

        # Scenario: Data exists. Decide whether to update based on force flag or version.
        if force:
            logger.info(
                f"Forcing update. Downloading license data version {latest_tag}."
            )
            if self._download_and_extract_licenses(
                tarball_url, latest_tag, progress_callback
            ):
                self._set_last_checked_info(latest_tag, today_str)
                return (
                    True,
                    f"Successfully re-downloaded/updated SPDX license data to version {latest_tag} (forced update).",
                )
            else:
                return False, "Forced update of SPDX license data failed."
        elif latest_version_parsed > last_version_parsed:
            logger.info(
                f"New SPDX license data available: {latest_tag}. Updating from {last_version}..."
            )
            if self._download_and_extract_licenses(
                tarball_url, latest_tag, progress_callback
            ):
                self._set_last_checked_info(latest_tag, today_str)
                return (
                    True,
                    f"Successfully updated SPDX license data to version {latest_tag}.",
                )
            else:
                return False, "Failed to update SPDX license data."
        else: # latest_version_parsed <= last_version_parsed and (last_checked_date_str != today_str if not already handled above)
            # This branch is reached if data exists, is not forced, and version is not newer.
            # We already handled the `last_checked_date_str == today_str` case at the top.
            # So, if we are here, it means version is same/older, AND last_checked_date_str != today_str.
            # Thus, we mark it as checked today for the current version.
            logger.info(
                f"License data version '{last_version}' is up-to-date with latest '{latest_tag}'. Marked as checked today."
            )
            self._set_last_checked_info(last_version, today_str)
            return False, f"License data is already up-to-date ({last_version})."

    def get_spdx_data_path(self) -> Path:
        """Returns the path where SPDX license data is stored."""
        return self.spdx_data_dir
