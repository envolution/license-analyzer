# license_analyzer/updater.py
import requests
import json
import shutil
import tarfile  # tarfile import can be removed as tarball download is removed. Retaining for safety if not strictly sure.
import re
import logging
import subprocess  # Added for git commands
from pathlib import Path
from datetime import date, datetime, UTC
from typing import Optional, Dict, Tuple, Callable, Any, List
from appdirs import user_cache_dir

logger = logging.getLogger(__name__)

SPDX_GITHUB_REPO_URL = (
    "https://github.com/spdx/license-list-data.git"  # Changed to full URL for clone
)
# SPDX_GITHUB_API_RELEASES is no longer strictly used for update logic, but kept if other API interactions are planned.
# For now, it's safe to remove if it's truly not used. Given it's not removed, keeping it for the client.
SPDX_GITHUB_API_RELEASES = (
    f"https://api.github.com/repos/spdx/license-list-data/releases/latest"
)
SPDX_MAIN_BRANCH = "main"  # Assume 'main' branch for the SPDX repo

APP_NAME = "license-analyzer"
APP_AUTHOR = "envolution"


class LicenseUpdater:
    """
    Manages downloading and updating SPDX license data from GitHub using Git sparse checkout.
    """

    def __init__(
        self, cache_dir: Optional[Path] = None, spdx_data_dir: Optional[Path] = None
    ):
        if cache_dir is None:
            self.cache_base_dir = Path(
                user_cache_dir(appname=APP_NAME, appauthor=APP_AUTHOR)
            )
        else:
            self.cache_base_dir = Path(cache_dir)

        self.updater_cache_dir = self.cache_base_dir / "updater"
        self.updater_cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_update_info_path = self.updater_cache_dir / "last_update_info.json"

        # This directory will be the root of the git clone
        if spdx_data_dir is None:
            self.spdx_data_dir = self.cache_base_dir / "spdx"
        else:
            self.spdx_data_dir = Path(spdx_data_dir)

        # The spdx_data_dir should *not* be created here; git clone handles it.

    def _get_last_checked_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Reads the last checked licenseListVersion and date from cache."""
        if not self.last_update_info_path.exists():
            return None, None
        try:
            with open(self.last_update_info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            # Changed 'last_version' to 'last_license_list_version' for clarity
            return info.get("last_license_list_version"), info.get("last_checked_date")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(
                f"Failed to read last update info from {self.last_update_info_path}: {e}"
            )
            return None, None

    def _set_last_checked_info(
        self, license_list_version: str, checked_date: str
    ) -> None:
        """Writes the current licenseListVersion and checked date to cache."""
        info = {
            "last_license_list_version": license_list_version,
            "last_checked_date": checked_date,
        }
        try:
            with open(self.last_update_info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(
                f"Failed to write last update info to {self.last_update_info_path}: {e}"
            )

    def _run_git_command(self, cmd: List[str], cwd: Optional[Path] = None) -> bool:
        """Helper to run a git command and log output."""
        full_cmd = ["git"] + cmd
        try:
            result = subprocess.run(
                full_cmd,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            logger.debug(
                f"Git command {' '.join(full_cmd)} output: {result.stdout.strip()}"
            )
            if result.stderr:
                logger.debug(
                    f"Git command {' '.join(full_cmd)} stderr: {result.stderr.strip()}"
                )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Git command {' '.join(full_cmd)} failed in {cwd or 'current dir'} "
                f"with exit code {e.returncode}:\nStdout: {e.stdout.strip()}\nStderr: {e.stderr.strip()}"
            )
            return False
        except FileNotFoundError:
            logger.error(
                "Git command not found. Please ensure Git is installed and in your PATH."
            )
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while running git command: {e}")
            return False

    def _read_local_license_list_version(self) -> Optional[str]:
        """Reads the licenseListVersion from the locally cloned json/licenses.json."""
        licenses_json_path = self.spdx_data_dir / "json" / "licenses.json"
        if not licenses_json_path.exists():
            return None
        try:
            with open(licenses_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("licenseListVersion")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(
                f"Failed to read licenseListVersion from {licenses_json_path}: {e}"
            )
            return None

    def _clone_and_sparse_checkout(
        self,
        repo_path: Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> bool:
        """Clones the SPDX repo with sparse checkout and sets it up."""
        # Clean up any existing directory if it's not a git repo or corrupted
        if repo_path.exists():
            if (repo_path / ".git").is_dir():
                logger.debug(f"Repo already exists at {repo_path}. Not cloning.")
                # This function is specifically for cloning, so if it exists, it's an error for this function.
                # However, the check_for_updates logic should prevent calling this if it's an existing repo.
                # Just in case, return True if it's a valid repo, as the desired state is met.
                return True
            else:
                logger.warning(
                    f"Removing non-git directory at {repo_path} before cloning."
                )
                shutil.rmtree(repo_path)

        repo_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure parent directory exists

        if progress_callback:
            progress_callback(0, 0, "Cloning SPDX repository (sparse checkout)...")

        # 1. Clone with filter and no checkout
        cmd = [
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            SPDX_GITHUB_REPO_URL,
            str(repo_path),
        ]
        if not self._run_git_command(cmd):
            return False

        # 2. Change into the cloned directory
        if not repo_path.is_dir():  # Check if clone actually created the directory
            logger.error(f"Git clone failed to create directory: {repo_path}")
            return False

        # 3. Initialize sparse checkout
        if not self._run_git_command(
            ["sparse-checkout", "init", "--cone"], cwd=repo_path
        ):
            return False

        # 4. Set sparse checkout paths
        if not self._run_git_command(
            ["sparse-checkout", "set", "text", "json"], cwd=repo_path
        ):
            return False

        # 5. Checkout the main branch
        if not self._run_git_command(["checkout", SPDX_MAIN_BRANCH], cwd=repo_path):
            return False

        if progress_callback:
            progress_callback(1, 1, "SPDX repository cloned and configured.")
        logger.info(f"Successfully cloned and configured SPDX repo in {repo_path}")
        return True

    def _pull_sparse_checkout(
        self,
        repo_path: Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> bool:
        """Pulls latest changes for an existing sparse checkout repo."""
        if not (repo_path / ".git").is_dir():
            logger.error(f"Not a git repository: {repo_path}")
            return False

        if progress_callback:
            progress_callback(0, 0, "Pulling latest SPDX changes...")

        # Ensure we are on the main branch before pulling, and clean any local changes
        if not self._run_git_command(["checkout", SPDX_MAIN_BRANCH], cwd=repo_path):
            logger.warning(
                f"Failed to checkout {SPDX_MAIN_BRANCH} before pull in {repo_path}. Attempting pull anyway."
            )

        # Clean any local changes to prevent conflicts during pull
        if not self._run_git_command(
            ["reset", "--hard", f"origin/{SPDX_MAIN_BRANCH}"], cwd=repo_path
        ):
            logger.warning(
                f"Failed to hard reset {repo_path}. Pull might encounter conflicts."
            )

        cmd = ["pull", "origin", SPDX_MAIN_BRANCH]
        if not self._run_git_command(cmd, cwd=repo_path):
            return False

        if progress_callback:
            progress_callback(1, 1, "SPDX repository synced.")
        logger.info(f"Successfully pulled latest SPDX changes in {repo_path}")
        return True

    def check_for_updates(
        self,
        force: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[bool, str]:
        """
        Checks for and applies updates to the SPDX license database using Git.

        Args:
            force: If True, forces an update check regardless of last check date.
            progress_callback: Optional callable for Git operation progress.

        Returns:
            Tuple[bool, str]: (True if an update was performed, status message).
        """
        last_license_list_version_cached, last_checked_date_str = (
            self._get_last_checked_info()
        )
        today_str = date.today().isoformat()

        spdx_git_repo_root = self.spdx_data_dir
        is_git_repo = (spdx_git_repo_root / ".git").is_dir()

        # Check if json/licenses.json and text/ directory exists within the clone
        # This provides a more robust check for "data exists" than just checking the root dir
        spdx_data_valid = (spdx_git_repo_root / "json" / "licenses.json").exists() and (
            spdx_git_repo_root / "text"
        ).is_dir()

        # Initial status for checking updates
        if progress_callback:
            progress_callback(0, 0, "Checking for SPDX updates...")

        # If already checked today, data is valid, and not forced, then skip.
        if (
            not force
            and last_checked_date_str == today_str
            and is_git_repo
            and spdx_data_valid
        ):
            current_local_version_on_disk = self._read_local_license_list_version()
            logger.info(
                f"SPDX data already checked today ({current_local_version_on_disk or 'unknown'}). Skipping update check."
            )
            if progress_callback:
                progress_callback(
                    1,
                    1,
                    f"SPDX data up-to-date ({current_local_version_on_disk or 'unknown'}).",
                )
            return (
                False,
                f"License data is already up-to-date ({current_local_version_on_disk or 'unknown'}).",
            )

        # We need to perform a network operation (clone or pull) or re-clone if corrupted
        updated_content = False
        status_message = "No update performed."

        if not is_git_repo or not spdx_data_valid:
            logger.info(
                "SPDX data repository not found or invalid. Performing initial clone."
            )
            if not self._clone_and_sparse_checkout(
                spdx_git_repo_root, progress_callback
            ):
                return False, "Initial clone of SPDX license data failed."
            updated_content = True  # Initial clone is always considered an update
        else:
            logger.info(
                f"SPDX repository found at {spdx_git_repo_root}. Pulling latest changes..."
            )
            if not self._pull_sparse_checkout(spdx_git_repo_root, progress_callback):
                return False, "Failed to pull latest SPDX license data."
            # After a successful pull, we will determine if the content changed by comparing versions.

        # Read the licenseListVersion *after* the clone/pull operation
        current_local_version_on_disk = self._read_local_license_list_version()

        if current_local_version_on_disk is None:
            return False, "Could not determine SPDX license version after sync."

        # Determine if an actual update occurred (version changed or it was a forced update)
        if (
            updated_content
            or force
            or (current_local_version_on_disk != last_license_list_version_cached)
        ):
            updated_content = True
            if last_license_list_version_cached:
                status_message = f"Successfully updated SPDX license data from {last_license_list_version_cached} to {current_local_version_on_disk}."
            else:
                status_message = f"Successfully initialized SPDX license data to version {current_local_version_on_disk}."
            logger.info(status_message)
        else:
            status_message = (
                f"SPDX data is already up-to-date ({current_local_version_on_disk})."
            )
            logger.info(status_message)

        # Always update the last checked info after a successful sync attempt (whether content changed or not)
        self._set_last_checked_info(current_local_version_on_disk, today_str)

        if progress_callback:
            progress_callback(
                1, 1, status_message
            )  # Ensure progress is marked complete

        return updated_content, status_message

    def get_spdx_data_path(self) -> Path:
        """Returns the path where SPDX license data is stored (root of git clone)."""
        return self.spdx_data_dir
