# nzachapy

> **Context Filtering for LLMs** — Scan your entire dataset and return only the portion relevant to your query.

When working with large datasets and language models, stuffing everything into the context window is noisy and expensive. `nzachapy` solves this by chunking your data, embedding it semantically, and returning only the chunks most relevant to a given query — so your model sees signal, not noise.

---

## Installation

```bash
pip install nzachapy
```

---

## Quick Start

```python
from nzachapy import Nzacha

# Initialize the module
nz = Nzacha()

# Add your dataset (must be a string)
nz.add_by_words(data)

# Query for relevant context
results = nz.query("your query here")
```

> ⚠️ **Important:** Your dataset must be converted to a string before passing it to `add_by_words()`.

---

## Configuration

When initializing `Nzacha`, you can customize the following parameters. If left unset, the defaults are used.

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `200` | Number of words per chunk |
| `overlap` | `20` | Number of overlapping words between consecutive chunks |
| `openai_api_key` | `None` | OpenAI API key for semantic embeddings (optional) |
| `threshold` | `0.6` | Minimum similarity score for a chunk to be returned |

### Example with custom config

```python
nz = Nzacha(
    chunk_size=150,
    overlap=30,
    openai_api_key="sk-...",
    threshold=0.75
)
```

---

## Embedding Backends

`nzachapy` supports two embedding strategies:

- **OpenAI** — Provide your `openai_api_key` at initialization to use OpenAI's embedding models for higher-quality semantic search.
- **Sentence Transformers** *(default)* — If no API key is provided, the module falls back to a local `sentence-transformers` model. No API key required.

---

## How It Works

1. **Chunking** — `add_by_words(data)` splits your string dataset into overlapping word-based chunks. `chunk_size` controls how many words each chunk contains; `overlap` controls how many words are shared between adjacent chunks to preserve context across boundaries.

2. **Embedding** — Each chunk is embedded into a vector using either OpenAI or Sentence Transformers.

3. **Retrieval** — When you run a query, it is embedded the same way and compared against all chunk embeddings. Only chunks that meet or exceed the `threshold` similarity score are returned.

---

## API Reference

### `Nzacha(chunk_size, overlap, openai_api_key, threshold)`
Initializes the context filter with the given configuration.

### `nz.add_by_words(data: str)`
Chunks the provided string dataset by word count and stores the embeddings internally. Must be called before querying.

- `data` — Your full dataset as a **string**. Convert lists, dicts, DataFrames, or any other structure to a string first.

---

## Notes

- Always pass data as a plain string. Use `str()`, `json.dumps()`, or `.to_string()` (for DataFrames) to convert before calling `add_by_words()`.
- A lower `threshold` returns more (but less precise) chunks. A higher threshold is stricter.
- Chunk `overlap` helps avoid losing context that falls on chunk boundaries — tune it based on your data's structure.

---

## Contributing

Contributions are welcome! If you spot an area for improvement — whether it's a new chunking strategy, a better embedding backend, performance optimizations, or additional retrieval methods — feel free to open an issue or submit a pull request.

Please make sure any changes are well-tested and clearly documented.

---

## License

MIT