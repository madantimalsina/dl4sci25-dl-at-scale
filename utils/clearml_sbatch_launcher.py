import os, sys, subprocess, json
from pathlib import Path

# Creates four parameter sections in ClearML UI: slurm.*, paths.*, git.*, train.* (+ env.*).
# Renders the sbatch from those values, uploads both the rendered sbatch and the template as artifacts.
# Records the SLURM job id back into the task (easy to track/resubmit).
# Next time we just clone the task in ClearML Web, edit slurm.num_nodes / slurm.gpus_per_node, click Enqueue. No code edits.

# --- helpers ---
def kv_overrides(argv):
    """Allow optional key=val overrides from CLI, but won't need them once UI is set."""
    out = {}
    for tok in argv:
        if "=" in tok and not tok.strip().startswith("--"):
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def render_template(tmpl: str, mapping: dict) -> str:
    return tmpl.format(**mapping)

def main():
    # ---- sensible defaults (used only on first run; after that the UI holds values) ----
    slurm = {
        "account":        "nstaff",
        "queue":          "regular",
        "num_nodes":      "2",      # <— change in UI later
        "gpus_per_node":  "4",      # <— change in UI later
        "cpus_per_task":  "32",
        "time_limit":     "01:00:00",
        "image":          "nersc/pytorch:24.08.01",
        "reservation":    "",           # "dl4sci" if any e.g dl4sci
        "job_name":       "vit-era5-ddp-clearml",
    }
    paths = {
        # "data_dir": "/pscratch/sd/s/shas1693/data/dl-at-scale-training-data",
        "data_dir": "/mscratch/sd/d/dasml/sc24_tutorial_data",
        # "log_path": os.path.join(os.environ.get("SCRATCH", "/tmp"), "dl-at-scale-training", "logs"),
        # "run_dir":  "/mscratch/sd/m/madan12/ClearML/clearml_logs/${SLURM_JOB_ID}",
    }
    git = {
        "repo":   "https://github.com/madantimalsina/dl4sci25-dl-at-scale.git",
        "branch": "clearml_test",
    }
    train = {
        # Everything after the wrapper (passed to train.py)
        "args": "--yaml_config config/ViT.yaml --config bs64_opt --num_data_workers 8 --expdir ./logs",
    }
    env = {
        "omp_threads": "4",
        "clearml_project": "dl4sci25-dl-at-scale",
        "task_name": "vit-multinode",
    }

    # ---- bind these dicts to the ClearML task so the Web UI stores them and re-enqueue ----
    task = None
    try:
        from clearml import Task, Logger
        task = Task.current_task() or Task.init(project_name=env["clearml_project"], task_name=env["task_name"])
        slurm  = task.connect(slurm,  name="slurm")   # appears as "slurm.*" in UI
        paths  = task.connect(paths,  name="paths")
        git    = task.connect(git,    name="git")
        train  = task.connect(train,  name="train")
        env    = task.connect(env,    name="env")
    except Exception:
        pass  # still works without ClearML, but we want the UI :)

    # Optional CLI key=val overrides (rarely needed once UI is the source of truth)
    overrides = kv_overrides(sys.argv[1:])
    for d in (slurm, paths, git, train, env):
        for k in list(d.keys()):
            if k in overrides: d[k] = overrides[k]

    # ---- load sbatch template and render with merged params ----
    template_path = Path(__file__).with_name("perlmutter_ddp_sbatch_template.sh")
    tmpl = template_path.read_text()

    # Build mapping for .format()
    mapping = {
        # slurm
        "account":        slurm["account"],
        "queue":          slurm["queue"],
        "num_nodes":      slurm["num_nodes"],
        "gpus_per_node":  slurm["gpus_per_node"],
        "cpus_per_task":  slurm["cpus_per_task"],
        "time_limit":     slurm["time_limit"],
        "image":          slurm["image"],
        "reservation":    slurm["reservation"],
        "job_name":       slurm["job_name"],
        # paths
        "data_dir":       paths["data_dir"],
        # "log_path":       paths["log_path"],
        # "run_dir":        paths["run_dir"],
        # git
        "repo":           git["repo"],
        "branch":         git["branch"],
        # train
        "train_args":     train["args"],
        # env
        "omp_threads":    env["omp_threads"],
        "clearml_project":env["clearml_project"],
        "task_name":      env["task_name"],
    }

    rendered = render_template(tmpl, mapping)

    # ---- write artifacts & submit ----
    out_dir = Path.cwd()
    out_path = out_dir / f"rendered_{slurm['job_name']}_{slurm['num_nodes']}x{slurm['gpus_per_node']}.sh"
    # Path(paths["log_path"]).mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered)
    print(f"[clearml] Rendered sbatch: {out_path}")

    if task:
        # store everything for perfect reproducibility + easy re-enqueue
        task.upload_artifact("sbatch_rendered", artifact_object=str(out_path))
        task.upload_artifact("sbatch_template", artifact_object=str(template_path))
        task.connect({"rendered_path": str(out_path)}, name="artifacts")  # visible path in UI
        Logger.current_logger().report_text("[sbatch]\n" + rendered[:5000])  # preview in logs

    # Submit
    print("[clearml] Submitting with sbatch…")
    p = subprocess.run(["sbatch", str(out_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout.strip())

    # Try to capture SLURM job id (common format: "Submitted batch job <id>")
    if task:
        line = p.stdout.strip()
        job_id = line.split()[-1] if line.endswith(tuple("0123456789")) else ""
        task.connect({"job_id": job_id, "submit_stdout": line}, name="slurm")

    sys.exit(p.returncode)

if __name__ == "__main__":
    main()
