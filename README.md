# Text Embedder

`text_embedder` is a powerful and flexible Python library for generating and managing text embeddings using pre-trained transformer based multilingual embedding models. It offers support for various pooling strategies, similarity functions, and quantization techniques, making it a versatile tool for a variety of NLP tasks, including embedding, similarity search, clustering, and more.

## 🚀 Features

- **Model Integration**: Wraps around 🤗 transformers to leverage the state-of-ther-art pre-trained embedding models.
- **Pooling Strategies**: Choose from multiple pooling methods such as CLS token, max/mean pooling, and more to tailor to your need.
- **Flexible Similarity Metrics**: Compute similarity scores between embeddings using cosine, dot, euclidean, and manhattan metrics.
- **Quantization Support**: Reduce memory usage and improve performance by quantizing embeddings to multiple precision levels with support for **auto mixed precision quantization**.
- **Prompt Support**: Optionally include a custom prompt in embeddings for contextualized representation.
- **Configurable Options**: Tune embedding generation with options for batch size, sequence length, normalization, and more.

## 🛠 Installation

Install `text_embedder` from PyPI using pip:

```bash
pip install text_embedder
```

## 📖 Usage

### Initialization

Initialize the `TransformersEmbedder` with your desired configurations:

```python
from text_embedder import TextEmbedder

embedder = TextEmbedder(
    model="BAAI/bge-small-en",
    sim_fn="cosine",
    pooling_strategy=["cls"],
    device="cuda",  # Specify device if needed
    precision="int8"  # Optional: for quantization
)
```

### Generating Embeddings

Generate embeddings for a list of texts:

```python
embeddings = embedder.embed(["Hello world", "Transformers are amazing!"])
print(embeddings)
```

### Computing Similarity

Compute similarity between two embeddings:

```python
embedding1 = embedder.embed(["Cat jumped from a chair"])
embedding2 = embedder.embed(["Mamba architecture is better than transformers tho, ngl."])
similarity_score = embedder.get_similarity(embedding1, embedding2)
print(f"Similarity Score: {similarity_score}")
```

## Advanced Usage

### Pooling Strategies

You can choose from various pooling strategies:
- `"cls"`: Use the CLS token embedding.
- `"max"`: Take the maximum value across tokens.
- `"mean"`: Compute the mean of token embeddings.
- `"mean_sqrt_len"`: Compute the mean divided by the square root of token length.
- `"weightedmean"`: Compute a weighted mean of token embeddings.
- `"lasttoken"`: Use the last token embedding.

### Similarity Functions

Supported similarity functions:
- **Cosine Similarity**: Measures the cosine of the angle between two vectors.
- **Dot Product**: Measures the dot product between two vectors.
- **Euclidean Distance**: Measures the straight-line distance between two vectors. (L1)
- **Manhattan Distance**: Measures the sum of absolute differences between two vectors. (L2)

### Quantization

Quantize embeddings to lower precision:
- **float32**: 32-bit floating-point precision.
- **float16**: 16-bit floating-point precision.
- **int8**: 8-bit integer precision.
- **uint8**: 8-bit unsigned integer precision.
- **binary**: Binary quantization.
- **ubinary**: Unsigned binary quantization.
- **2bit**: 2-bit quantization.
- **4bit**: 4-bit quantization.
- **8bit**: 8-bit quantization.

### Future Work
- Additional Pooling Strategies: Implement more advanced pooling methods (eg., attention-based). Also have to add a `auto` option to pooling_strategy to find a right pooling method based on model config.
- Custom Quantization Methods: Add to new quantization techniques for further improvement.
- Similarity function: Also add more similarity metric functions

## 🤝 Contributing

Contributions are welcome! Please follow these steps to get started with your contribution:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature/your-feature`).
6. Create a new Pull Request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/xdevfaheem/transformers_embedder/blob/main/LICENSE) file for details.

## Acknowledgement

Special Thanks to devs of [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers/tree/master) library. 