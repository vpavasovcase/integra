# Pydantic Model

Simple example of using PydanticAI to construct a Pydantic model from a text input.

Demonstrates:

* [structured `result_type`](../results.md#structured-result-validation)

## Running the Example

With [dependencies installed and environment variables set](./index.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.pydantic_model
```

This examples uses `openai:gpt-4o` by default, but it works well with other models, e.g. you can run it
with Gemini using:

```bash
PYDANTIC_AI_MODEL=gemini-1.5-pro python/uv-run -m pydantic_ai_examples.pydantic_model
```

(or `PYDANTIC_AI_MODEL=gemini-1.5-flash ...`)

## Example Code

```python {title="pydantic_model.py"}
#! examples/pydantic_ai_examples/pydantic_model.py
```
