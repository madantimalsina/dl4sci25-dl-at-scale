# utils/ddp_multinode_clearml.py
import os, sys
def maybe_init_clearml():
    try:
        from clearml import Task
    except Exception:
        return
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    if rank == 0 and (Task.current_task() is None):
        Task.init(
            project_name=os.environ.get("CLEARML_PROJECT", "dl4sci25-dl-at-scale"),
            task_name=os.environ.get("CLEARML_TASK_NAME", "vit-multinode"),
        )

def main():
    maybe_init_clearml()
    os.execvp(sys.executable, [sys.executable, "train.py", *sys.argv[1:]])

if __name__ == "__main__":
    main()
