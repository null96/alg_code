# alg_code

Minimal target repository for validating the OpenClaw local-execution workflow.

## Entry point

After bootstrap, the worker can run:

```bash
python -m automl_runner --phase smoke --data-dir <LOCAL_PATH>
```

This package is intentionally tiny and dependency-free so the bridge can validate:

- bootstrap succeeds
- the repo installs with `pip install -e .`
- `run_task` can execute a committed module entrypoint
- worker results return to relay and OpenClaw
