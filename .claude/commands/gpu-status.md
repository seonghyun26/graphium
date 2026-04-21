Check what processes are running on each GPU and print a summary table.

For each GPU, show:
- GPU index
- Memory used / total
- PID of the running process (if any)
- The model name (extract `model=XXX` from the process command line)
- The task/dataset (extract `tasks=XXX` from the process command line)
- Status: "idle" if no process, "running" otherwise

Run these bash commands to gather the information:

1. `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader` to get GPU memory usage
2. For each GPU with a running process, use `nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader` to map PIDs to GPUs, then read `/proc/<pid>/cmdline` to extract the model and tasks.

Print the result as a clean markdown table with columns: GPU, Mem Used, Mem Total, PID, Model, Tasks, Status.
