# LLM-From-Scratch

A 162M parameter language model built from scratch in PyTorch and trained on an RTX 4060 Laptop GPU on 4 billion tokens from FineWeb.

**Final validation loss: 3.494** (perplexity [how surprised the model is by text, lower is better] of about 32.9)




## Sample Outputs

Unedited samples from the final model. Settings: temperature 0.9, top-k 40, 48 new tokens, so the text stops mid-sentence.

**Prompt:** `The Amazon River is a river in south america that`

> The Amazon River is a river in south america that flows into the Gulf of Mexico in the north american. It has a population of approximately 5,500 and is a natural reservoir of the bayside estuaries. It is rich with freshwater and was the center of the bays…

**Prompt:** `The steps to make toast are`

> The steps to make toast are as follows:
> - First, remove the muffin pan from the oven.
> - In a small bowl, combine the eggs, one at a time, with the vanilla and two tablespoons of sugar.
> - Sprinkle with the baking…


The text reads fluently, but the facts are often wrong (the Amazon does not flow into the Gulf of Mexico) and it drifts off topic (toast turns into a muffin recipe). That is normal for a model this size.




## Model

GPT-2 style, with these settings:

- 12 layers, embedding size 768, 16 attention heads
- Context length 512 (how many tokens it reads at once), vocab size 50,257 (GPT-2 tokenizer)
- 162,694,225 parameters

GPT-2 small has 124M parameters because it uses weight tying [the input embedding and the output layer share the same weights]. This model keeps them separate, which adds about 38.6M parameters. With tying it would be about 124M.



## Training

- **Data:** FineWeb `sample-10BT`, tokenized with the GPT-2 tokenizer (tiktoken). 99% train, 1% validation.
- **Batch:** 4 sequences x 512 tokens x 8 gradient accumulation steps [adding up gradients over several small batches before each update] = 16,384 tokens per update.
- **Optimizer:** AdamW with gradient clipping at 1.0. The learning rate warms up, then decays down to 1e-5.
- **Hardware:** RTX 4060 Laptop GPU in fp32 (full precision), about 4,300 tokens per second.

Validation loss is checked on 20 batches (about 41,000 tokens), so it is a little noisy.




## Files

- `gpt_model.py`: tokenizer, model, text generator and dataset classes
- `train_gpt.ipynb`: data prep, training loop and loss plots



## Limitations

This is a base model [trained only to predict the next word, not to follow instructions or chat]. It will ramble, repeat itself and make things up.

## Credits

Trained on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb). Tokenizer from [tiktoken](https://github.com/openai/tiktoken).

