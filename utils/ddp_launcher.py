# ddp_launcher.py
import os
import sys
import shlex
import subprocess

RESERVED_KEYS = {"ngpus", "nproc_per_node", "master_port", "port"}

def parse_key_value_args(argv):
    """
    From ClearML we receive args like: ["yaml_config=config/ViT.yaml", "config=bs16_opt", "ngpus=4"]
    Return:
      - params: dict of key->value for key=value tokens
      - passthrough: list of tokens that were not key=value (e.g., raw --flags)
    """
    params = {}
    passthrough = []
    for tok in argv:
        if "=" in tok and not tok.startswith("--"):
            k, v = tok.split("=", 1)
            params[k.strip()] = v.strip()
        else:
            # allow raw CLI style tokens to pass through unchanged
            passthrough.append(tok)
    return params, passthrough

def to_cli_flags(params):
    """
    Convert {"yaml_config":"config/ViT.yaml","config":"bs16_opt"} -> ["--yaml_config","config/ViT.yaml","--config","bs16_opt"]
    """
    cli = []
    for k, v in params.items():
        cli.extend([f"--{k}", v])
    return cli

def main():
    # 1) Parse incoming args
    params, passthrough = parse_key_value_args(sys.argv[1:])

    # 2) Figure out number of GPUs
    ngpus = (
        int(params.get("ngpus") or params.get("nproc_per_node")
            or os.environ.get("NGPUS", "0"))
        if any(k in params for k in ("ngpus", "nproc_per_node")) or "NGPUS" in os.environ
        else 0
    )
    if ngpus <= 0:
        # fallback to CUDA device count if not explicitly set
        try:
            import torch
            ngpus = torch.cuda.device_count()
        except Exception:
            ngpus = 1  # last resort

    # 3) Optional port override
    port = int(params.get("master_port") or params.get("port") or os.environ.get("MASTER_PORT", "29500"))

    # 4) Build train.py CLI (remove reserved keys from the param dict)
    train_params = {k: v for k, v in params.items() if k not in RESERVED_KEYS}
    train_cli = to_cli_flags(train_params) + passthrough

    # 5) Compose torch.distributed.run command (single-node)
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={ngpus}",
        f"--master_port={port}",
        "train.py",
    ] + train_cli

    print("Launching DDP:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
