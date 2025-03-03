# Fine-tuning GPT-2 para Tradução Inglês-Yanomami

# Este script configura o processo de fine-tuning do modelo GPT-2 small (124M) para tradução
# entre inglês e Yanomami, com foco em funcionamento offline.

# Etapas:
# 1. Carregamento e preparação do dataset
# 2. Configuração do modelo GPT-2
# 3. Treinamento manual do modelo
# 4. Avaliação do modelo
# 5. Salvamento para uso offline

import os
import json
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.model_selection import train_test_split
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

dataset_path = '/Users/renanserrano/CascadeProjects/Yanomami/finetunning/yanomami_dataset/'
dataset_files = [
    'translations.jsonl',
    'yanomami-to-english.jsonl',
    'phrases.jsonl',
    'grammar.jsonl',
    'comparison.jsonl',
    'how-to.jsonl'
]

all_data = []

for file in dataset_files:
    try:
        file_data = load_jsonl(os.path.join(dataset_path, file))
        all_data.extend(file_data)
        print(f"Carregado {len(file_data)} exemplos de {file}")
    except Exception as e:
        print(f"Erro ao carregar {file}: {e}")

print(f"Total de exemplos carregados: {len(all_data)}")

print("\nAmostra dos dados:")
print(all_data[:2])

# Verificar a estrutura dos dados carregados
if all_data:
    print(f"Estrutura do primeiro exemplo: {all_data[0]}")
else:
    print("Nenhum dado carregado.")

def prepare_data_for_training(examples):
    processed_data = []
    for example in examples:
        # Verificar a estrutura do exemplo
        if 'messages' in example:
            for message in example['messages']:
                if message['role'] == 'user' and 'content' in message:
                    user_input = message['content']
                elif message['role'] == 'assistant' and 'content' in message:
                    assistant_output = message['content']
                    # Adicionar ao processed_data
                    processed_data.append({"english": user_input, "yanomami": assistant_output})
    return processed_data

processed_data = prepare_data_for_training(all_data)
print(f"Total de exemplos processados: {len(processed_data)}")

train_data, val_data = train_test_split(processed_data, test_size=0.1, random_state=42)
print(f"Exemplos de treinamento: {len(train_data)}")
print(f"Exemplos de validação: {len(val_data)}")

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# 3. Configuração do modelo GPT-2

model_name = "gpt2"  

# Verificar se o modelo já existe localmente para suporte offline
model_output_dir = "./gpt2_yanomami_translator"
if os.path.exists(model_output_dir) and os.path.isdir(model_output_dir):
    print(f"Carregando modelo e tokenizer de {model_output_dir}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_output_dir)
    model = GPT2LMHeadModel.from_pretrained(model_output_dir)
else:
    print(f"Carregando modelo e tokenizer pré-treinados de {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Configurar o dispositivo para treinamento
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
model.to(device)

def tokenize_function(examples):
    # Inicializa listas para armazenar os dados processados
    english_texts = []
    yanomami_texts = []
    
    # Processa cada exemplo no lote
    # Os dados já estão no formato {'english': texto, 'yanomami': texto}
    english_texts = examples['english']
    yanomami_texts = examples['yanomami']
    
    # Combina os textos para tokenização
    combined_texts = [f"{eng} => {yan}" for eng, yan in zip(english_texts, yanomami_texts)]
    
    # Retorna um dicionário vazio se não houver textos para processar
    if not combined_texts:
        print("No valid examples found in batch.")
        return {"input_ids": [], "attention_mask": []}
    
    # Tokeniza os textos combinados
    tokenized = tokenizer(combined_texts, padding="max_length", truncation=True, max_length=128)
    
    return tokenized

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask"])
tokenized_val.set_format("torch", columns=["input_ids", "attention_mask"])

# 4. Treinamento do modelo

# Definir hiperparâmetros para treinamento manual
num_epochs = 3
batch_size = 4
learning_rate = 5e-5
warmup_steps = 50
max_grad_norm = 1.0

class GPT2Dataset(TorchDataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        # Obter o exemplo do dataset
        example = self.encodings[idx]
        # Criar um dicionário com input_ids e attention_mask
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "labels": example["input_ids"].clone()
        }

train_dataset = GPT2Dataset(tokenized_train)
val_dataset = GPT2Dataset(tokenized_val)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Configurar otimizador
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Configurar scheduler
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# Função de treinamento manual
def train_model():
    # Colocar o modelo em modo de treinamento
    model.train()
    total_train_loss = 0
    
    # Treinar por número definido de epochs
    for epoch in range(num_epochs):
        logger.info(f"Iniciando epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        
        # Treinar em batches
        for batch_idx, batch in enumerate(train_dataloader):
            # Mover batch para o dispositivo
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Zerar gradientes
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Clipping de gradiente
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Atualizar pesos
            optimizer.step()
            scheduler.step()
            
            # Acumular perda
            epoch_loss += loss.item()
            
            # Log a cada 10 batches
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Calcular perda média da epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        total_train_loss += avg_epoch_loss
        logger.info(f"Epoch {epoch+1} concluída. Perda média: {avg_epoch_loss:.4f}")
        
        # Avaliar no conjunto de validação
        eval_loss = evaluate_model()
        logger.info(f"Perda de validação: {eval_loss:.4f}")
    
    # Calcular perda média do treinamento
    avg_train_loss = total_train_loss / num_epochs
    logger.info(f"Treinamento concluído. Perda média de treinamento: {avg_train_loss:.4f}")
    return avg_train_loss

# Função de avaliação
def evaluate_model():
    # Colocar o modelo em modo de avaliação
    model.eval()
    total_eval_loss = 0
    
    # Avaliar sem calcular gradientes
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_eval_loss += loss.item()
    
    # Calcular perda média
    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    return avg_eval_loss

# Executar treinamento
logger.info("Iniciando treinamento...")
try:
    avg_train_loss = train_model()
    logger.info(f"Treinamento concluído com sucesso. Perda média: {avg_train_loss:.4f}")
except Exception as e:
    logger.error(f"Erro durante o treinamento: {e}")

# 5. Avaliação do modelo

# Avaliar o modelo final
final_eval_loss = evaluate_model()
perplexity = np.exp(final_eval_loss)
logger.info(f"Avaliação final - Perda: {final_eval_loss:.4f}, Perplexidade: {perplexity:.2f}")

def generate_translation(text, max_length=100):
    inputs = tokenizer(text, return_tensors="pt")
    # Suporte para diferentes dispositivos (CUDA, MPS, CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)
    
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

test_texts = [
    "Traduzir de inglês para Yanomami: Hello, how are you? =>",
    "Traduzir de Yanomami para inglês: Kami yanomae thë ã =>",
]

print("\nTestes de tradução:")
for text in test_texts:
    translation = generate_translation(text)
    print(f"Entrada: {text}")
    print(f"Saída: {translation}\n")

# 6. Salvamento para uso offline

output_dir = "./gpt2_yanomami_translator"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Modelo e tokenizer salvos em: {output_dir}")

# Criar um arquivo README.md com instruções de uso
readme_content = """
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
"""

with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)
print("Arquivo README.md criado com instruções de uso")

# 7. Código para usar o modelo offline

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def load_yanomami_translator(model_path):
    """Carrega o modelo e tokenizer para tradução Yanomami-Inglês.
    
    Args:
        model_path (str): Caminho para o diretório contendo o modelo e tokenizer
        
    Returns:
        tuple: (model, tokenizer) carregados e prontos para uso
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"O diretório do modelo {model_path} não existe.")
        
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Configurar o dispositivo para inferência
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    model.to(device)
    print(f"Modelo carregado no dispositivo: {device}")
    
    return model, tokenizer

# Exemplo de uso
if __name__ == "__main__" and os.path.exists("./gpt2_yanomami_translator"):
    print("\n\n===== TESTANDO O MODELO TREINADO =====\n")
    try:
        model_path = "./gpt2_yanomami_translator"
        test_model, test_tokenizer = load_yanomami_translator(model_path)
        
        # Exemplos de tradução
        test_phrases = [
            "What does 'aheprariyo' mean in Yanomami?",
            "How do you say 'hello' in Yanomami?"
        ]
        
        for phrase in test_phrases:
            print(f"\nEntrada: {phrase}")
            translation = generate_translation(phrase)
            print(f"Tradução: {translation}")
            
    except Exception as e:
        print(f"Erro ao testar o modelo: {e}")

# # Testar algumas traduções
# test_texts = [
#     "Traduzir de inglês para Yanomami: Hello, how are you? =>",
#     "Traduzir de Yanomami para inglês: Kami yanomae thë ã =>",
# ]
# 
# print("\nTestes de tradução:")
# for text in test_texts:
#     translation = generate_translation(text)
#     print(f"Entrada: {text}")
#     print(f"Saída: {translation}\n")
