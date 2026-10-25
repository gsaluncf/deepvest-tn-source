"""Install and start an isolated Java Unity Catalog service using Docker Desktop."""
import argparse
import json
import subprocess
import time
import urllib.request

IMAGE = "unitycatalog/unitycatalog@sha256:91758d439edd0f9cb51727700ff8e0817e2f4ff02334cd66d9a431dabcaa457e"
LABEL = "tn.submission.unity"


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


def wait_ready(url, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/2.1/unity-catalog/catalogs?max_results=1", timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, TimeoutError):
            time.sleep(0.5)
    raise RuntimeError(f"Unity did not become ready at {url}")


def install(name="tn-submission-unity", port=18080):
    if not 1 <= port <= 65535:
        raise ValueError("Unity port must be between 1 and 65535")
    docker("info", "--format", "{{.ServerVersion}}")
    existing = docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}")
    if existing:
        raise RuntimeError(f"Container {name} already exists. Use --unity-url to reuse it or choose another --name.")
    docker("pull", IMAGE)
    workdir = docker("image", "inspect", IMAGE, "--format", "{{.Config.WorkingDir}}")
    if not workdir.startswith("/") or workdir == "/":
        raise RuntimeError("Unity image has no application working directory")
    container = docker("run", "-d", "--name", name, "--label", LABEL + "=true",
                       "-p", f"127.0.0.1:{port}:8080", "--mount",
                       f"type=volume,source={name}-data,target={workdir}/etc/db", IMAGE,
                       "sh", "-c", "sed -i 's/^server.authorization=.*/server.authorization=disable/' etc/conf/server.properties; exec bin/start-uc-server")
    url = f"http://127.0.0.1:{port}"
    try:
        wait_ready(url)
    except Exception:
        docker("stop", container)
        raise
    return {"container": container, "name": name, "url": url, "image": IMAGE}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="tn-submission-unity")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    print(json.dumps(install(args.name, args.port), indent=2))
