"""RunPod REST v1 driver (deploy / status / terminate). Key from repo .env, Bearer auth."""
import json, sys, urllib.request, urllib.error

import os; ENV = os.environ.get("REEFSCAN_ENV", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), ".env"))
KEY = next((l.split("=", 1)[1].strip().strip('"').strip("'")
            for l in open(ENV) if l.startswith("RUNPOD_API_KEY=")), None)
BASE = "https://rest.runpod.io/v1"


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            txt = r.read().decode()
            return json.loads(txt) if txt.strip() else {}
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:600]); sys.exit(3)


def deploy(gpu, pubkey, cloud):
    setup = ("apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server "
             ">/dev/null 2>&1; mkdir -p /run/sshd /root/.ssh; "
             f"echo {pubkey} >> /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys; "
             "/usr/sbin/sshd -D")
    body = {
        "name": "reefscan-cpp", "imageName": "nvcr.io/nvidia/pytorch:24.10-py3",
        "gpuTypeIds": [gpu], "gpuCount": 1, "containerDiskInGb": 45, "volumeInGb": 0,
        "cloudType": cloud, "supportPublicIp": True, "ports": ["22/tcp"],
        "dockerStartCmd": ["bash", "-lc", setup],
    }
    d = call("POST", "/pods", body)
    print(d.get("id", json.dumps(d)))


def status(pod_id):
    d = call("GET", f"/pods/{pod_id}")
    out = {"status": d.get("desiredStatus"),
           "ports": d.get("portMappings") or d.get("ports"),
           "publicIp": d.get("publicIp"),
           "runtime_ports": (d.get("runtime") or {}).get("ports")}
    print(json.dumps(out))


def terminate(pod_id):
    call("DELETE", f"/pods/{pod_id}")
    print("terminated", pod_id)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "deploy": deploy(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "SECURE")
    elif c == "status": status(sys.argv[2])
    elif c == "terminate": terminate(sys.argv[2])
