# Development

## Documentation

Install and run the documentation toolchain through the `docs` dependency group:

```bash
uv run --group docs zensical serve
uv run --group docs zensical build --clean
```

The Python API reference is rendered by mkdocstrings from runtime PyO3
docstrings, so the docs build uses `--group docs` rather than `--only-group
docs` to make the local `noobase` package importable.

## Tests

```bash
cargo test --workspace
uv run pytest crates/noobase-py/tests/
```

## Lints

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```
