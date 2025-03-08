# SQL Generation

Example demonstrating how to use PydanticAI to generate SQL queries based on user input.

Demonstrates:

* [dynamic system prompt](../agents.md#system-prompts)
* [structured `result_type`](../results.md#structured-result-validation)
* [result validation](../results.md#result-validators-functions)
* [agent dependencies](../dependencies.md)

## Running the Example

The resulting SQL is validated by running it as an `EXPLAIN` query on PostgreSQL. To run the example, you first need to run PostgreSQL, e.g. via Docker:

```bash
docker run --rm -e POSTGRES_PASSWORD=postgres -p 54320:5432 postgres
```
_(we run postgres on port `54320` to avoid conflicts with any other postgres instances you may have running)_

With [dependencies installed and environment variables set](./index.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.sql_gen
```

or to use a custom prompt:

```bash
python/uv-run -m pydantic_ai_examples.sql_gen "find me errors"
```

This model uses `gemini-1.5-flash` by default since Gemini is good at single shot queries of this kind.

## Example Code

```python {title="sql_gen.py"}
#! examples/pydantic_ai_examples/sql_gen.py
```
