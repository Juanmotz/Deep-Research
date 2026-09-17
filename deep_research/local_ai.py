from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_MODEL_ROOTS = [
    Path.cwd(),
    Path(r"F:\"),
    Path.home(),
]


def _normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve() if str(path).strip() else Path.cwd()


def discover_model_files(base_roots: list[str | Path] | None = None) -> list[Path]:
    roots = [_normalize_path(item) for item in (base_roots or DEFAULT_MODEL_ROOTS)]
    discovered: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for match in root.rglob("*.gguf"):
            if match.is_file():
                discovered.add(match.resolve())
    return sorted(discovered, key=lambda p: str(p).lower())


def discover_server_binaries(base_roots: list[str | Path] | None = None) -> list[Path]:
    roots = [_normalize_path(item) for item in (base_roots or DEFAULT_MODEL_ROOTS)]
    discovered: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for match in root.rglob("llama-server.exe"):
            if match.is_file():
                discovered.add(match.resolve())
        for match in root.rglob("llama-server"):
            if match.is_file():
                discovered.add(match.resolve())
    return sorted(discovered, key=lambda p: str(p).lower())


def _server_command(binary: Path, model_path: Path, port: int) -> list[str]:
    base = [str(binary), "-m", str(model_path), "--host", "127.0.0.1", "--port", str(port)]
    if platform.system().lower() == "windows":
        base.extend(["-ngl", "999"])
    return base


def wait_for_server(port: int, timeout: float = 45.0) -> bool:
    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            request = Request(health_url, headers={"User-Agent": "DeepResearchLocalAI/1.0"})
            with urlopen(request, timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def start_local_server(model_path: str | Path | None = None, port: int = 8080, server_binary: str | Path | None = None) -> dict:
    model_candidates = discover_model_files()
    if model_path is None:
        if not model_candidates:
            raise FileNotFoundError("No .gguf model files were found. Put a model in the project, F:\\, or a nearby directory.")
        model_path = model_candidates[0]
    model_path = _normalize_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    server_candidates = discover_server_binaries()
    if server_binary is None:
        if not server_candidates:
            raise FileNotFoundError("No llama-server binary was found. Place llama-server.exe in a bin directory or provide --server-binary.")
        server_binary = server_candidates[0]
    server_binary = _normalize_path(server_binary)
    if not server_binary.exists():
        raise FileNotFoundError(f"Model server binary not found: {server_binary}")

    command = _server_command(server_binary, model_path, port)
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ready = wait_for_server(port)
    if not ready:
        process.terminate()
        raise RuntimeError(f"The local model server did not become ready on port {port} within the timeout.")

    return {
        "model_path": str(model_path),
        "server_binary": str(server_binary),
        "port": port,
        "pid": process.pid,
        "health_url": f"http://127.0.0.1:{port}/health",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch a local GGUF-backed model server similar to the portable F drive setup.")
    parser.add_argument("--model", help="Path to a .gguf model file. Defaults to the first discovered model.")
    parser.add_argument("--server-binary", help="Path to llama-server.exe or llama-server. Defaults to the first discovered server binary.")
    parser.add_argument("--port", type=int, default=8080, help="Port to expose the local API on.")
    parser.add_argument("--list-models", action="store_true", help="List discovered GGUF models and exit.")
    parser.add_argument("--list-servers", action="store_true", help="List discovered llama-server binaries and exit.")
    args = parser.parse_args(argv)

    if args.list_models:
        models = discover_model_files()
        if not models:
            print("No GGUF model files were found.")
            return 0
        for model in models:
            print(model)
        return 0

    if args.list_servers:
        servers = discover_server_binaries()
        if not servers:
            print("No llama-server binaries were found.")
            return 0
        for server in servers:
            print(server)
        return 0

    try:
        details = start_local_server(model_path=args.model, port=args.port, server_binary=args.server_binary)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("Local model server started successfully.")
    print(f"Model: {details['model_path']}")
    print(f"Server: {details['server_binary']}")
    print(f"Health: {details['health_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
