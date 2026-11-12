"""Offline-first evaluation harness for the comparison pipeline (issue 5).

`python -m evals.run --llm fake` runs every fixture in evals/fixtures/
end-to-end with no network call and no API key, computing the metrics in
evals/metrics.py. `--llm do_serverless` / `--llm claude_cli` (evals/llm_backends.py)
exercise a real model and need credentials set as environment variables;
see that module's docstring.
"""
