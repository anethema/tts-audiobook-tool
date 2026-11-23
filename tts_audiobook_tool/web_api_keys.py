"""Save browser LLM credentials under the current Windows user's DPAPI profile."""

from base64 import b64decode, b64encode
from hashlib import sha256
from ipaddress import ip_address
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from urllib.parse import urlsplit

from tts_audiobook_tool.constants import APP_NAME


class StoredApiKeyError(ValueError):
    """An API key storage error suitable for display in browser controls."""


_KEY_PATTERN = re.compile(r"[A-Za-z0-9._~+/=-]+")
_FILE_ERROR = "Could not access saved API key storage."
_DPAPI_ERROR = "Could not encrypt or unlock the saved API key with Windows protection."
_POWERSHELL_LOCATION = Path("System32/WindowsPowerShell/v1.0/powershell.exe")
_DPAPI_SCRIPT = """
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Security
$endpoint = [Convert]::FromBase64String([Console]::ReadLine())
$data = [Convert]::FromBase64String([Console]::ReadLine())
$scope = [System.Security.Cryptography.DataProtectionScope]::CurrentUser
$result = [System.Security.Cryptography.ProtectedData]::{method}($data, $endpoint, $scope)
[Console]::Write([Convert]::ToBase64String($result))
"""
_DPAPI_COMMANDS = {
    method: b64encode(_DPAPI_SCRIPT.format(method=method).encode("utf-16-le")).decode("ascii")
    for method in ("Protect", "Unprotect")
}


def _is_loopback(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def _endpoint_digest(endpoint: str, key: str) -> bytes:
    invalid_endpoint = "Enter a full HTTP or HTTPS chat completions URL without credentials or a fragment."
    if not isinstance(endpoint, str):
        raise StoredApiKeyError(invalid_endpoint)
    endpoint = endpoint.strip()
    if key and key in endpoint:
        raise StoredApiKeyError("Remove the API key from the endpoint URL before saving.")
    if "#" in endpoint or "\\" in endpoint or any(
        char.isspace() or ord(char) < 32 or ord(char) == 127 for char in endpoint
    ):
        raise StoredApiKeyError(invalid_endpoint)
    try:
        address = urlsplit(endpoint)
        hostname = address.hostname
        if (address.scheme not in {"http", "https"} or not hostname or address.username is not None
                or address.port == 0):
            raise ValueError
        raw_endpoint = endpoint.encode("utf-8")
    except ValueError:
        raise StoredApiKeyError(invalid_endpoint) from None
    if address.scheme == "http" and not _is_loopback(hostname):
        raise StoredApiKeyError("Use HTTPS to save an API key for a non-loopback endpoint.")
    return sha256(raw_endpoint).digest()


def _storage_directory() -> Path:
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        root = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    else:
        root = Path.home() / ".local" / "share"
    return root / APP_NAME / "web-api-keys"


def _key_location(endpoint: str, *, create: bool = False, key: str = "") -> tuple[Path, bytes] | None:
    try:
        digest = _endpoint_digest(endpoint, key)
    except StoredApiKeyError:
        if create:
            raise
        return None

    try:
        directory = _storage_directory()
        if create:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or (
            os.name != "nt" and (info.st_uid != os.getuid() or info.st_mode & 0o077)
        ):
            raise StoredApiKeyError(_FILE_ERROR)
    except FileNotFoundError:
        if create:
            raise StoredApiKeyError(_FILE_ERROR) from None
        return None
    except (OSError, RuntimeError):
        raise StoredApiKeyError(_FILE_ERROR) from None
    return directory / (digest.hex() + ".dpapi"), digest


def _windows_powershell() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("SystemRoot", r"C:\Windows")) / _POWERSHELL_LOCATION
    # WSL can invoke Windows PowerShell directly from the mounted system drive.
    return Path("/mnt/c/Windows") / _POWERSHELL_LOCATION


def _run_dpapi(method: str, digest: bytes, data: bytes) -> bytes:
    try:
        result = subprocess.run(
            [str(_windows_powershell()), "-NoLogo", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
             "-EncodedCommand", _DPAPI_COMMANDS[method]],
            input=b64encode(digest) + b"\n" + b64encode(data) + b"\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            timeout=10,
            check=True,
        )
        if not result.stdout:
            raise ValueError
        return b64decode(result.stdout, validate=True)
    except (OSError, subprocess.SubprocessError, ValueError):
        raise StoredApiKeyError(_DPAPI_ERROR) from None


def save_api_key(endpoint: str, key: str) -> None:
    """Encrypt the pasted key and replace any saved key for the exact URL."""
    key = key.strip() if isinstance(key, str) else ""
    if not _KEY_PATTERN.fullmatch(key):
        raise StoredApiKeyError("Paste a valid API key before saving.")
    location = _key_location(endpoint, create=True, key=key)
    assert location is not None
    path, digest = location
    ciphertext = _run_dpapi("Protect", digest, key.encode("ascii"))

    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                temporary.write(ciphertext)
                temporary.close()
                temporary_path.replace(path)
            finally:
                temporary.close()
                temporary_path.unlink(missing_ok=True)
    except OSError:
        raise StoredApiKeyError(_FILE_ERROR) from None


def load_api_key(endpoint: str) -> str:
    """Unlock the key for the exact URL, or return an empty string when absent."""
    location = _key_location(endpoint)
    if location is None:
        return ""
    path, digest = location
    try:
        ciphertext = path.read_bytes()
    except FileNotFoundError:
        return ""
    except OSError:
        raise StoredApiKeyError(_FILE_ERROR) from None
    plaintext = _run_dpapi("Unprotect", digest, ciphertext)
    try:
        key = plaintext.decode("ascii")
    except UnicodeError:
        raise StoredApiKeyError(_DPAPI_ERROR) from None
    if not _KEY_PATTERN.fullmatch(key):
        raise StoredApiKeyError(_DPAPI_ERROR)
    return key


def has_api_key(endpoint: str) -> bool:
    """Report saved ciphertext without unlocking or returning the key."""
    location = _key_location(endpoint)
    try:
        return location is not None and location[0].is_file()
    except OSError:
        raise StoredApiKeyError(_FILE_ERROR) from None


def forget_api_key(endpoint: str) -> None:
    """Remove the saved ciphertext for the exact URL, if present."""
    location = _key_location(endpoint)
    if location is None:
        return
    try:
        location[0].unlink(missing_ok=True)
    except OSError:
        raise StoredApiKeyError(_FILE_ERROR) from None
