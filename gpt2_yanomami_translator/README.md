
# Tradutor Yanomami-Inglês

Este é um modelo de tradução bidirecional entre Inglês e Yanomami baseado no GPT-2.

## Requisitos
- Python 3.6+
- PyTorch
- Transformers

## Como usar

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Carregar o modelo e tokenizer
model_path = "./gpt2_yanomami_translator"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Função para tradução
def translate(text, max_length=100):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Exemplos de uso
# Para traduzir do inglês para Yanomami:
print(translate("What does 'aheprariyo' mean in Yanomami?"))

# Para traduzir de Yanomami para inglês:
print(translate("aheprariyo"))
```

Este modelo foi projetado para funcionar totalmente offline, sem necessidade de conexão à internet.

## 🌐 Resources

- [Hugging Face Model](https://huggingface.co/renanserrano/yanomami-finetuning)