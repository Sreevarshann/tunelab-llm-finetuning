# Security

TuneLab is a **single-user, local** tool: the API has no authentication and binds to 127.0.0.1 (compose publishes on 127.0.0.1 only). Do not expose it to a network without adding auth.

- Uploads: size limit, UTF-8 only, strict JSON per line, schema validation, row and length limits; content is stored as data and never executed.
- Model output is displayed as text only (React escapes it), never executed or used as HTML.
- Paths: datasets and adapters live under server-generated ids; no user-supplied paths are opened.
- Jobs run in subprocesses with resource isolation from the API; cancellation falls back to killing the process group.
- Model weights come from a pinned Hugging Face revision; no remote code is executed (MLX loads safetensors).
- CI runs full-history gitleaks and `npm audit`.
