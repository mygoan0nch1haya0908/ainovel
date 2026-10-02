import json
import os
import sqlite3
import subprocess
import sys

import pytest

from ainovel.security import secret_vault

SENTINEL = "SENTINEL_KEY_42"


@pytest.fixture
def api():
    return secret_vault


def test_unsupported_platform_rejects_secret_storage(api, tmp_path, monkeypatch):
    monkeypatch.setattr(api.sys, "platform", "linux")
    with pytest.raises(api.VaultError):
        api.DpapiSecretVault(tmp_path / "credentials.db").put(SENTINEL)
    assert not (tmp_path / "credentials.db").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows current-user DPAPI integration")
def test_dpapi_restart_round_trip_ciphertext_and_delete(api, tmp_path):
    path = tmp_path / "private" / "credentials.db"
    vault = api.DpapiSecretVault(path)
    reference = vault.put(SENTINEL)
    assert SENTINEL.encode() not in path.read_bytes()
    # Verify actual Windows ACLs, including inheritance for future journal files.
    quoted = str(path).replace("'", "''")
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
        "$p='" + quoted + "'; $sid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value; "
        "@($p,(Split-Path -Parent $p)) | ForEach-Object { $a=Get-Acl -LiteralPath $_; "
        "[pscustomobject]@{Protected=$a.AreAccessRulesProtected; "
        "Rules=@($a.GetAccessRules($true,$true,[System.Security.Principal.SecurityIdentifier]) | "
        "ForEach-Object { $_.IdentityReference.Value }); Expected=$sid} } | ConvertTo-Json -Depth 4"],
        check=True, capture_output=True, text=True,
        env={key: value for key, value in os.environ.items() if key.lower() != "psmodulepath"})
    assert not result.stderr, result.stderr
    for acl in json.loads(result.stdout):
        assert acl["Protected"]
        assert acl["Rules"] == [acl["Expected"]]
    restarted = api.DpapiSecretVault(path)
    assert restarted.get(reference) == SENTINEL
    restarted.delete(reference)
    with pytest.raises(api.VaultError) as error:
        restarted.get(reference)
    assert SENTINEL not in str(error.value)
    reference = restarted.put(SENTINEL)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE credentials SET ciphertext=? WHERE reference=?", (b"corrupt", reference))
    with pytest.raises(api.VaultError):
        restarted.get(reference)


def test_failed_protection_blocks_persistence(api, tmp_path, monkeypatch):
    path = tmp_path / "private" / "credentials.db"
    def fail(*args):
        raise RuntimeError(SENTINEL)
    monkeypatch.setattr(api, "_protect_path", fail)
    with pytest.raises(api.VaultError) as error:
        api.DpapiSecretVault(path).put(SENTINEL)
    assert SENTINEL not in str(error.value)
    assert not path.exists()


def test_encryption_failure_is_fixed_and_leaves_no_record(api, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError(SENTINEL)
    monkeypatch.setattr(api, "_encrypt", fail)
    path = tmp_path / "private" / "credentials.db"
    with pytest.raises(api.VaultError) as error:
        api.DpapiSecretVault(path).put(SENTINEL)
    assert SENTINEL not in str(error.value)
    assert not path.exists()
