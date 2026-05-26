# alg_code

Minimal target repository for validating the OpenClaw local-execution workflow.

## Entry point

After bootstrap, the worker can run:

```bash
python -m automl_runner --phase inspect --data-dir <LOCAL_PATH>
python -m automl_runner --phase smoke --data-dir <LOCAL_PATH>
```

This package is intentionally small so the bridge can validate:

- bootstrap succeeds
- the repo installs with `pip install -e .`
- `run_task` can execute a committed module entrypoint
- worker results return to relay and OpenClaw
- a simple AF classification smoke run can complete on local CSV data
