# Yanomami Language Dataset

This dataset was created for training the Yanomami-English translation model. It contains various types of data to help the model learn different aspects of the Yanomami language and its translation to English.

## Dataset Contents

The dataset consists of the following files:

| File | Description | Examples |
|------|-------------|----------|
| translations.jsonl | General translations between Yanomami and English | 17,009 |
| yanomami-to-english.jsonl | Specific Yanomami to English translations | 1,822 |
| phrases.jsonl | Common phrases and expressions | 2,322 |
| grammar.jsonl | Grammatical structures and rules | 200 |
| comparison.jsonl | Comparative phrases and structures | 2,072 |
| how-to.jsonl | Instructional content and procedural language | 5,586 |

## Data Format

Each file is in JSONL format (JSON Lines), where each line is a valid JSON object containing the source text and its translation.

Example format:
```json
{"input": "English: What does 'aheprariyo' mean in Yanomami? => Yanomami:", "output": "aheprariyo means 'to be happy' in Yanomami."}
{"input": "Yanomami: ahetoimi => English:", "output": "ahetoimi means 'to be sick' in English."}
```

## Dataset Generation

This dataset was generated using the [AI Dataset Generator](https://www.npmjs.com/package/ai-dataset-generator) npm package, which helps create synthetic training data for language models.

## Related Resources

### Models & Repositories
- **Fine-tuned Model (Hugging Face)**: [renanserrano/yanomami-finetuning](https://huggingface.co/renanserrano/yanomami-finetuning)
- **Dataset (Hugging Face)**: [renanserrano/yanomami](https://huggingface.co/datasets/renanserrano/yanomami)
- **GitHub Repository**: [renantrendt/yanomami-finetuning](https://github.com/renantrendt/yanomami-finetuning)
- **Dataset Generator (GitHub)**: [renantrendt/ai-dataset-generator](https://github.com/renantrendt/ai-dataset-generator)

## Usage

This dataset is used to fine-tune the GPT-2 model for Yanomami-English translation. The diverse nature of the dataset helps the model learn various aspects of the language, including vocabulary, grammar, and common expressions.

## License

This dataset is provided for research and educational purposes. Please respect the Yanomami culture and language when using this data.

## Citation

If you use this dataset in your research or applications, please cite:

```
@misc{yanomami-english-dataset,
  author = {Renan Serrano},
  title = {Yanomami Language Dataset},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/renantrendt/yanomami-finetuning}}
}
```
