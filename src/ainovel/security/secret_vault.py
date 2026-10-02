"""Current-user Windows DPAPI vault, separate from the application database.

The protected directory also restricts newly created SQLite sidecars. Revocation
cannot erase old backups or recall requests already sent. DPAPI is tied to the
Windows account; this is not a portable backup format or protection from processes
running as that user. Python cannot promise complete plaintext memory erasure.
"""

import ctypes
from ctypes import wintypes
from pathlib import Path
import sqlite3
import sys
from typing import Protocol
from uuid import UUID, uuid4

DEFAULT_VAULT_PATH = Path("D:/ainovel/.worktrees/qwen-adapter/.superpowers/runtime/model-profiles/credentials.db")
MAX_SECRET_CHARS = 8192
MAX_SECRET_BYTES = MAX_SECRET_CHARS * 4
MAX_CIPHERTEXT_BYTES = 65536


class VaultError(ValueError):
    def __init__(self):
        super().__init__("credential storage unavailable")


class SecretVault(Protocol):
    def put(self, secret: str) -> str: ...
    def get(self, reference: str) -> str: ...
    def delete(self, reference: str) -> None: ...


class _Blob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _windows():
    if sys.platform != "win32":
        raise VaultError()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    return kernel


def _crypt(data: bytes, *, decrypt: bool) -> bytes:
    kernel = _windows()
    bound = MAX_CIPHERTEXT_BYTES if decrypt else MAX_SECRET_BYTES
    if not data or len(data) > bound:
        raise VaultError()
    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    source = _Blob(len(data), buffer)
    result = _Blob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    function = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    function.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
    function.restype = wintypes.BOOL
    try:
        # CRYPTPROTECT_UI_FORBIDDEN; deliberately no CRYPTPROTECT_LOCAL_MACHINE.
        if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
            raise VaultError()
        output_bound = MAX_SECRET_BYTES if decrypt else MAX_CIPHERTEXT_BYTES
        if not result.data or not 1 <= result.size <= output_bound:
            raise VaultError()
        return ctypes.string_at(result.data, result.size)
    finally:
        ctypes.memset(buffer, 0, len(data))
        if result.data:
            if result.size <= MAX_CIPHERTEXT_BYTES:
                ctypes.memset(result.data, 0, result.size)
            kernel.LocalFree(result.data)


def _encrypt(data: bytes) -> bytes:
    return _crypt(data, decrypt=False)


def _decrypt(data: bytes) -> bytes:
    return _crypt(data, decrypt=True)


def _protect_path(path: Path) -> None:
    """Replace inherited permissions with a protected current-user-only DACL."""
    kernel = _windows()
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                          wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    advapi.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    advapi.SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
    token = wintypes.HANDLE()
    sid_text = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    try:
        if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
            raise VaultError()
        size = wintypes.DWORD()
        advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
        if not 1 <= size.value <= 65536:
            raise VaultError()
        info = ctypes.create_string_buffer(size.value)
        if not advapi.GetTokenInformation(token, 1, info, size.value, ctypes.byref(size)):
            raise VaultError()
        sid = ctypes.cast(info, ctypes.POINTER(ctypes.c_void_p))[0]
        if not advapi.ConvertSidToStringSidW(sid, ctypes.byref(sid_text)):
            raise VaultError()
        sddl = f"D:P(A;OICI;FA;;;{ctypes.wstring_at(sid_text)})"
        if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ctypes.byref(descriptor), None):
            raise VaultError()
        if not advapi.SetFileSecurityW(str(path), 0x80000004, descriptor):
            raise VaultError()
    finally:
        if descriptor:
            kernel.LocalFree(descriptor)
        if sid_text:
            kernel.LocalFree(sid_text)
        if token:
            kernel.CloseHandle(token)


class DpapiSecretVault:
    def __init__(self, path: Path | str = DEFAULT_VAULT_PATH):
        self.path = Path(path).absolute()

    def _connect(self):
        _windows()
        # Do not redirect the credential boundary via symlinks/junctions.
        for path in (self.path, *self.path.parents):
            if path.is_symlink() or path.is_junction():
                raise VaultError()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _protect_path(self.path.parent)
        for suffix in ("", "-journal", "-wal", "-shm"):
            path = Path(str(self.path) + suffix)
            if path.exists():
                if path.is_symlink() or path.is_junction():
                    raise VaultError()
                _protect_path(path)
        connection = sqlite3.connect(self.path)
        try:
            _protect_path(self.path)
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA secure_delete=ON")
            connection.execute("CREATE TABLE IF NOT EXISTS credentials (reference TEXT PRIMARY KEY, ciphertext BLOB NOT NULL)")
            return connection
        except Exception:
            connection.close()
            raise

    def put(self, secret: str) -> str:
        try:
            _windows()
            if not isinstance(secret, str) or not 1 <= len(secret) <= MAX_SECRET_CHARS:
                raise VaultError()
            ciphertext = _encrypt(secret.encode("utf-8"))
            reference = str(uuid4())
            connection = self._connect()
            try:
                with connection:
                    connection.execute("INSERT INTO credentials VALUES (?, ?)", (reference, ciphertext))
            finally:
                connection.close()
            return reference
        except Exception:
            raise VaultError() from None

    def get(self, reference: str) -> str:
        try:
            UUID(reference)
            connection = self._connect()
            try:
                row = connection.execute("SELECT ciphertext FROM credentials WHERE reference = ?", (reference,)).fetchone()
            finally:
                connection.close()
            if row is None:
                raise VaultError()
            secret = _decrypt(row[0]).decode("utf-8")
            if not 1 <= len(secret) <= MAX_SECRET_CHARS:
                raise VaultError()
            return secret
        except Exception:
            raise VaultError() from None

    def delete(self, reference: str) -> None:
        try:
            UUID(reference)
            connection = self._connect()
            try:
                with connection:
                    connection.execute("DELETE FROM credentials WHERE reference = ?", (reference,))
            finally:
                connection.close()
        except Exception:
            raise VaultError() from None
