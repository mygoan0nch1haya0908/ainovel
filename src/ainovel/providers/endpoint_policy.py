"""Syntax-only endpoint normalization. Saving never performs DNS or networking."""

from dataclasses import dataclass
import ipaddress
import re
from urllib.parse import urlsplit, urlunsplit


class EndpointError(ValueError):
    def __init__(self):
        super().__init__("invalid model endpoint")


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    host: str
    port: int
    path: str
    kind: str


def normalize_endpoint(value: str, connection_kind: str) -> Endpoint:
    try:
        if (not isinstance(value, str) or not 1 <= len(value) <= 2048
                or connection_kind not in ("remote", "loopback")
                or any(ord(c) <= 32 or ord(c) == 127 for c in value)
                or any(c in value for c in "\\?#")):
            raise EndpointError()
        parsed = urlsplit(value)
        if (parsed.scheme not in ("http", "https") or not parsed.netloc
                or parsed.username is not None or parsed.password is not None
                or not parsed.hostname or "%" in parsed.netloc):
            raise EndpointError()
        if connection_kind == "remote" and parsed.scheme != "https":
            raise EndpointError()
        host = parsed.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
            if (len(host) > 253 or not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                                          for label in host.split("."))):
                raise EndpointError()
        if connection_kind == "loopback" and host != "localhost" and not (address and address.is_loopback):
            raise EndpointError()
        if address is not None:
            host = str(address)
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        if not 1 <= port <= 65535 or parsed.netloc.endswith(":"):
            raise EndpointError()
        path = parsed.path.rstrip("/")
        # Do not let alternate encodings or dot segments change the authorized base path.
        if any(segment in (".", "..") for segment in path.split("/")) or "%" in path:
            raise EndpointError()
        authority = f"[{host}]" if ":" in host else host
        if port != (443 if parsed.scheme == "https" else 80):
            authority += f":{port}"
        normalized = urlunsplit((parsed.scheme, authority, path, "", ""))
        return Endpoint(normalized, host, port, path, connection_kind)
    except Exception:
        raise EndpointError() from None
