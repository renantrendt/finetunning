#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Yanomami RAG Translator
-----------------------
Um sistema de tradução Yanomami-Inglês baseado em RAG (Retrieval-Augmented Generation)
que funciona totalmente offline.

Este sistema combina:
1. Um índice de busca vetorial para recuperar as traduções mais relevantes
2. Um modelo de geração de texto para formatar as respostas
"""

import os
import json
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YanomamiRAGTranslator:
    """Sistema de tradução Yanomami-Inglês baseado em RAG que funciona offline."""
    
    def __init__(self, 
                 dataset_dir, 
                 embedding_model_name='all-MiniLM-L6-v2',
                 gpt2_model_dir='gpt2_yanomami_translator',
                 index_path='yanomami_index.faiss',
                 entries_path='yanomami_entries.json',
                 device=None):
        """
        Inicializa o tradutor RAG Yanomami.
        
        Args:
            dataset_dir: Diretório contendo os arquivos JSONL do dataset
            embedding_model_name: Nome do modelo de embeddings a ser usado
            gpt2_model_dir: Diretório contendo o modelo GPT-2 fine-tuned
            index_path: Caminho para salvar/carregar o índice FAISS
            entries_path: Caminho para salvar/carregar as entradas do dataset
            device: Dispositivo para execução (None para autodetecção)
        """
        self.dataset_dir = dataset_dir
        self.embedding_model_name = embedding_model_name
        self.gpt2_model_dir = gpt2_model_dir
        self.index_path = index_path
        self.entries_path = entries_path
        
        # Determinar o dispositivo (CPU ou GPU)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Carregar modelos e dados
        self._load_embedding_model()
        self._load_gpt2_model()
        
        # Verificar se o índice e entradas existem, caso contrário, criar
        if os.path.exists(index_path) and os.path.exists(entries_path):
            self._load_index_and_entries()
        else:
            self._build_index_and_entries()
    
    def _load_embedding_model(self):
        """Carrega o modelo de embeddings."""
        logger.info(f"Carregando modelo de embeddings: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_model.to(self.device)
    
    def _load_gpt2_model(self):
        """Carrega o modelo GPT-2 fine-tuned."""
        logger.info(f"Carregando modelo GPT-2 de: {self.gpt2_model_dir}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.gpt2_model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(self.gpt2_model_dir)
        self.model.to(self.device)
        self.model.eval()  # Modo de avaliação
    
    def _load_index_and_entries(self):
        """Carrega o índice FAISS e as entradas do dataset."""
        logger.info(f"Carregando índice de: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        
        logger.info(f"Carregando entradas de: {self.entries_path}")
        with open(self.entries_path, 'r', encoding='utf-8') as f:
            self.all_entries = json.load(f)
    
    def _build_index_and_entries(self):
        """Constrói o índice FAISS e as entradas do dataset."""
        logger.info("Construindo índice e entradas do dataset...")
        
        # Carregar todos os arquivos JSONL do dataset
        all_entries = []
        dataset_files = [
            'translations.jsonl', 
            'yanomami-to-english.jsonl', 
            'phrases.jsonl',
            'grammar.jsonl', 
            'comparison.jsonl', 
            'how-to.jsonl'
        ]
        
        for file in dataset_files:
            file_path = os.path.join(self.dataset_dir, file)
            if not os.path.exists(file_path):
                logger.warning(f"Arquivo não encontrado: {file_path}")
                continue
                
            logger.info(f"Processando arquivo: {file}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        # Processar entradas no formato antigo
                        if 'english' in entry and 'yanomami' in entry:
                            query = entry['english']
                            all_entries.append({
                                'query': query,
                                'entry': {
                                    'messages': [
                                        {'role': 'user', 'content': query},
                                        {'role': 'assistant', 'content': entry['yanomami']}
                                    ]
                                }
                            })
                            
                            # Adicionar também na direção inversa
                            query_inverse = f"Translate to English: {entry['yanomami']}"
                            all_entries.append({
                                'query': query_inverse,
                                'entry': {
                                    'messages': [
                                        {'role': 'user', 'content': query_inverse},
                                        {'role': 'assistant', 'content': entry['english']}
                                    ]
                                }
                            })
                        
                        # Processar entradas no formato novo (com messages)
                        elif 'messages' in entry:
                            user_message = None
                            assistant_message = None
                            
                            for msg in entry['messages']:
                                if msg['role'] == 'user' and 'content' in msg:
                                    user_message = msg['content']
                                elif msg['role'] == 'assistant' and 'content' in msg:
                                    assistant_message = msg['content']
                            
                            if user_message and assistant_message:
                                all_entries.append({
                                    'query': user_message,
                                    'entry': entry
                                })
                    except json.JSONDecodeError:
                        logger.error(f"Erro ao decodificar JSON na linha: {line}")
                    except Exception as e:
                        logger.error(f"Erro ao processar entrada: {e}")
        
        self.all_entries = all_entries
        logger.info(f"Total de entradas processadas: {len(all_entries)}")
        
        # Criar embeddings
        logger.info("Criando embeddings para as consultas...")
        queries = [entry['query'] for entry in all_entries]
        
        # Processar em batches para evitar estouro de memória
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(queries), batch_size)):
            batch_queries = queries[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_queries, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        
        # Criar índice FAISS
        logger.info("Criando índice FAISS...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Salvar índice e entradas
        logger.info(f"Salvando índice em: {self.index_path}")
        faiss.write_index(self.index, self.index_path)
        
        logger.info(f"Salvando entradas em: {self.entries_path}")
        with open(self.entries_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_entries, f, ensure_ascii=False, indent=2)
    
    def _detect_language(self, text):
        """
        Detecta se o texto é em inglês ou yanomami.
        
        Args:
            text: Texto a ser analisado
            
        Returns:
            'english' ou 'yanomami'
        """
        # Caracteres comuns em Yanomami que são menos comuns em inglês
        yanomami_chars = ['ë', 'ï', 'ö', 'ü', 'ã', 'õ', 'ñ', 'ë', 'ï', 'ö', 'ü']
        
        # Palavras comuns em Yanomami
        yanomami_words = ['pë', 'thë', 'kë', 'ya', 'wa', 'wamaki', 'yamaki', 'pei', 'yai', 'a', 'ma']
        
        # Verificar caracteres especiais
        for char in yanomami_chars:
            if char in text.lower():
                return 'yanomami'
        
        # Verificar palavras comuns
        for word in yanomami_words:
            if f" {word} " in f" {text.lower()} ":
                return 'yanomami'
        
        # Se não encontrou características de Yanomami, provavelmente é inglês
        return 'english'
    
    def _parse_translation_query(self, query):
        """
        Analisa a consulta para determinar a direção da tradução e o texto a ser traduzido.
        
        Args:
            query: Consulta a ser analisada
            
        Returns:
            Tupla (direction, text_to_translate)
            direction: 'to_yanomami', 'to_english' ou 'unknown'
            text_to_translate: Texto a ser traduzido
        """
        query = query.strip()
        
        # Verificar padrões explícitos de tradução
        if query.lower().startswith("translate to yanomami:"):
            return 'to_yanomami', query[len("translate to yanomami:"):].strip()
        
        if query.lower().startswith("translate to english:"):
            return 'to_english', query[len("translate to english:"):].strip()
        
        if query.lower().startswith("translate:"):
            text = query[len("translate:"):].strip()
            lang = self._detect_language(text)
            return 'to_yanomami' if lang == 'english' else 'to_english', text
        
        if "yanomami" in query.lower() and "mean" in query.lower():
            # Extrair a palavra entre aspas simples ou duplas
            import re
            match = re.search(r"['\"]([^'\"]+)['\"]|\b(\w+)\b", query)
            if match:
                word = match.group(1) or match.group(2)
                return 'to_english', word
        
        # Tentar detectar o idioma automaticamente
        lang = self._detect_language(query)
        return 'to_english' if lang == 'yanomami' else 'to_yanomami', query
    
    def get_comprehensive_info(self, word_or_phrase, top_k=10):
        """
        Retorna informações completas sobre uma palavra ou frase, incluindo exemplos,
        frases, gramática e comparações de todos os arquivos de dataset.
        
        Args:
            word_or_phrase: Palavra ou frase para buscar informações
            top_k: Número de entradas mais similares a recuperar por categoria
            
        Returns:
            Dicionário com informações completas organizadas por categoria
        """
        # Detectar idioma da palavra ou frase
        language = self._detect_language(word_or_phrase)
        
        # Formatar consultas para diferentes tipos de busca
        if language == 'yanomami':
            base_query = f"Translate to English: {word_or_phrase}"
            meaning_query = f"What does '{word_or_phrase}' mean in Yanomami?"
        else:  # english
            base_query = f"Translate to Yanomami: {word_or_phrase}"
            meaning_query = f"How do you say '{word_or_phrase}' in Yanomami?"
        
        # Codificar as consultas
        base_embedding = self.embedding_model.encode([base_query])
        meaning_embedding = self.embedding_model.encode([meaning_query])
        
        # Buscar entradas mais similares para cada consulta
        base_distances, base_indices = self.index.search(
            np.array(base_embedding).astype('float32'), 
            top_k * 5  # Buscar mais entradas para ter mais opções de filtragem
        )
        
        meaning_distances, meaning_indices = self.index.search(
            np.array(meaning_embedding).astype('float32'), 
            top_k * 5  # Buscar mais entradas para ter mais opções de filtragem
        )
        
        # Combinar e remover duplicatas
        all_indices = list(set(base_indices[0].tolist() + meaning_indices[0].tolist()))
        
        # Recuperar as entradas correspondentes
        retrieved_entries = [self.all_entries[idx] for idx in all_indices]
        
        # Organizar entradas por tipo de arquivo
        categorized_entries = {
            'translations': [],
            'yanomami_to_english': [],
            'phrases': [],
            'grammar': [],
            'comparison': [],
            'how_to': [],
            'other': []
        }
        
        # Função para extrair o nome do arquivo da entrada
        def get_file_category(entry_query):
            if 'translate to yanomami' in entry_query.lower() or 'translate to english' in entry_query.lower():
                return 'translations'
            if 'what does' in entry_query.lower() and 'mean in yanomami' in entry_query.lower():
                return 'yanomami_to_english'
            if 'how do you say' in entry_query.lower() and 'in yanomami' in entry_query.lower():
                return 'yanomami_to_english'
            if 'phrase' in entry_query.lower() or 'sentence' in entry_query.lower():
                return 'phrases'
            if 'grammar' in entry_query.lower() or 'structure' in entry_query.lower():
                return 'grammar'
            if 'compare' in entry_query.lower() or 'difference' in entry_query.lower():
                return 'comparison'
            if 'how to' in entry_query.lower() or 'how do' in entry_query.lower():
                return 'how_to'
            return 'other'
        
        # Filtrar e categorizar entradas
        for entry in retrieved_entries:
            original_entry = entry['entry']
            user_message = ""
            assistant_message = ""
            
            for msg in original_entry['messages']:
                if msg['role'] == 'user' and 'content' in msg:
                    user_message = msg['content']
                elif msg['role'] == 'assistant' and 'content' in msg:
                    assistant_message = msg['content']
            
            # Verificar se a entrada contém a palavra ou frase buscada
            word_or_phrase_lower = word_or_phrase.lower()
            if word_or_phrase_lower in user_message.lower() or word_or_phrase_lower in assistant_message.lower():
                # Categorizar a entrada
                category = get_file_category(user_message)
                
                # Adicionar à categoria correspondente
                if len(categorized_entries[category]) < top_k:  # Limitar o número de entradas por categoria
                    categorized_entries[category].append({
                        'query': user_message,
                        'response': assistant_message
                    })
        
        # Gerar resposta formatada com as informações organizadas
        result = {
            'word_or_phrase': word_or_phrase,
            'language': language,
            'categories': {}
        }
        
        # Adicionar categorias não vazias ao resultado
        for category, entries in categorized_entries.items():
            if entries:
                result['categories'][category] = entries
        
        return result

    def translate(self, query, top_k=5, temperature=0.7, max_length=200, direct_translation=False, comprehensive=False):
        """
        Traduz uma consulta usando o sistema RAG.
        
        Args:
            query: Consulta a ser traduzida
            top_k: Número de entradas mais similares a recuperar
            temperature: Temperatura para geração de texto
            max_length: Comprimento máximo da resposta gerada
            direct_translation: Se True, tenta fazer uma tradução direta sem explicações adicionais
            comprehensive: Se True, retorna informações completas sobre a palavra ou frase
            
        Returns:
            Tradução gerada ou informações completas se comprehensive=True
        """
        # Se for solicitada informação completa, usar o método especializado
        if comprehensive:
            return self.get_comprehensive_info(query)
        
        # Analisar a consulta para determinar a direção da tradução
        direction, text_to_translate = self._parse_translation_query(query)
        
        # Verificar se é uma consulta de tradução direta
        is_direct_translation = direct_translation or query.lower().startswith("translate:") or \
                               query.lower().startswith("translate to yanomami:") or \
                               query.lower().startswith("translate to english:")
        
        # Verificar se é uma consulta de significado (o que significa esta palavra)
        is_meaning_query = "what does" in query.lower() and "mean" in query.lower() or \
                         "how do you say" in query.lower() and "in yanomami" in query.lower()
        
        # Para consultas de significado, usar o método comprehensive
        if is_meaning_query:
            comprehensive_info = self.get_comprehensive_info(text_to_translate)
            return self._format_comprehensive_info(comprehensive_info)
        
        # Reformatar a consulta para melhorar a recuperação
        if direction == 'to_yanomami':
            formatted_query = f"Translate to Yanomami: {text_to_translate}"
        else:  # to_english ou unknown
            formatted_query = f"Translate to English: {text_to_translate}"
        
        # Codificar a consulta formatada
        query_embedding = self.embedding_model.encode([formatted_query])
        
        # Buscar entradas mais similares
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            top_k * 2  # Buscar mais entradas para ter mais opções de filtragem
        )
        
        # Recuperar as entradas correspondentes
        retrieved_entries = [self.all_entries[idx] for idx in indices[0]]
        
        # Filtrar entradas para priorizar a direção da tradução correta
        filtered_entries = []
        for entry in retrieved_entries:
            original_entry = entry['entry']
            user_message = ""
            assistant_message = ""
            
            for msg in original_entry['messages']:
                if msg['role'] == 'user' and 'content' in msg:
                    user_message = msg['content']
                elif msg['role'] == 'assistant' and 'content' in msg:
                    assistant_message = msg['content']
            
            # Verificar se a direção da tradução corresponde
            if direction == 'to_yanomami' and 'translate to yanomami' in user_message.lower():
                filtered_entries.append((original_entry, 10))  # Peso maior para direção exata
            elif direction == 'to_english' and 'translate to english' in user_message.lower():
                filtered_entries.append((original_entry, 10))  # Peso maior para direção exata
            # Verificar se há palavras semelhantes no texto a ser traduzido
            elif text_to_translate.lower() in user_message.lower() or any(word in user_message.lower() for word in text_to_translate.lower().split() if len(word) > 3):
                filtered_entries.append((original_entry, 5))  # Peso médio para conteúdo semelhante
            else:
                filtered_entries.append((original_entry, 1))  # Peso normal para outras entradas
        
        # Ordenar por peso (prioridade)
        filtered_entries.sort(key=lambda x: x[1], reverse=True)
        
        # Limitar ao número de exemplos desejados
        filtered_entries = filtered_entries[:top_k]
        
        # Construir contexto para o modelo
        context = "Com base nas seguintes informações:\n\n"
        
        # Adicionar instruções específicas para a direção da tradução
        if direction == 'to_yanomami':
            context += "Você é um tradutor especializado em traduzir de Inglês para Yanomami.\n"
            if is_direct_translation:
                context += "Traduza diretamente o texto para Yanomami, sem explicações adicionais.\n\n"
            else:
                context += "Traduza o texto para Yanomami, mantendo o significado original.\n\n"
        else:  # to_english ou unknown
            context += "Você é um tradutor especializado em traduzir de Yanomami para Inglês.\n"
            if is_direct_translation:
                context += "Traduza diretamente o texto para Inglês, sem explicações adicionais.\n\n"
            else:
                context += "Traduza o texto para Inglês, mantendo o significado original.\n\n"
        
        # Adicionar exemplos recuperados
        for i, (entry, weight) in enumerate(filtered_entries):
            user_message = ""
            assistant_message = ""
            
            for msg in entry['messages']:
                if msg['role'] == 'user' and 'content' in msg:
                    user_message = msg['content']
                elif msg['role'] == 'assistant' and 'content' in msg:
                    assistant_message = msg['content']
            
            context += f"[Exemplo {i+1}]\nConsulta: {user_message}\nTradução: {assistant_message}\n\n"
        
        # Gerar resposta usando o modelo fine-tuned com o contexto
        if is_direct_translation:
            prompt = f"{context}\nConsulta: {formatted_query}\nTradução (apenas a tradução direta, sem explicações):"
        else:
            prompt = f"{context}\nConsulta: {formatted_query}\nTradução:"
        
        # Tokenizar e gerar resposta
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_length=len(inputs['input_ids'][0]) + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extrair apenas a parte relevante da resposta
        if "Tradução:" in response:
            response = response.split("Tradução:")[-1].strip()
        elif "Consulta:" in response and formatted_query in response:
            # Tentar extrair a resposta após a consulta
            parts = response.split(formatted_query)
            if len(parts) > 1:
                response = parts[-1].strip()
                # Remover qualquer texto após a próxima consulta, se houver
                if "Consulta:" in response:
                    response = response.split("Consulta:")[0].strip()
        
        # Limpar a resposta de qualquer texto que não seja a tradução
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            # Ignorar linhas que parecem ser parte do prompt ou instruções
            if line.startswith('[Exemplo') or 'Consulta:' in line or 'Tradução:' in line:
                continue
            if 'Yanomami' in line and ('traduzir' in line.lower() or 'especializado' in line.lower()):
                continue
            if 'Inglês' in line and ('traduzir' in line.lower() or 'especializado' in line.lower()):
                continue
            cleaned_lines.append(line)
        
        response = '\n'.join(cleaned_lines).strip()
        
        # Se a resposta for exatamente igual à consulta original, tentar gerar novamente com temperatura maior
        if response == text_to_translate and temperature < 0.9:
            return self.translate(query, top_k=top_k+2, temperature=0.9, max_length=max_length)
        
        return response
    
    def _format_comprehensive_info(self, info):
        """
        Formata as informações completas de forma legível.
        
        Args:
            info: Dicionário com informações completas retornado por get_comprehensive_info
            
        Returns:
            String formatada com as informações completas
        """
        word_or_phrase = info['word_or_phrase']
        language = info['language']
        categories = info['categories']
        
        # Iniciar com um cabeçalho
        if language == 'yanomami':
            result = f"## Informações sobre '{word_or_phrase}' (Yanomami)\n\n"
        else:
            result = f"## Informações sobre '{word_or_phrase}' (Inglês)\n\n"
        
        # Adicionar tradução direta se disponível
        if 'translations' in categories and categories['translations']:
            if language == 'yanomami':
                result += "### Tradução para Inglês\n"
            else:
                result += "### Tradução para Yanomami\n"
                
            for entry in categories['translations'][:3]:  # Mostrar até 3 traduções
                result += f"- {entry['response']}\n"
            result += "\n"
        
        # Adicionar significado se disponível
        if 'yanomami_to_english' in categories and categories['yanomami_to_english']:
            result += "### Significado\n"
            for entry in categories['yanomami_to_english'][:3]:  # Mostrar até 3 significados
                result += f"- {entry['response']}\n"
            result += "\n"
        
        # Adicionar frases de exemplo se disponíveis
        if 'phrases' in categories and categories['phrases']:
            result += "### Frases de Exemplo\n"
            for entry in categories['phrases'][:5]:  # Mostrar até 5 frases
                result += f"- **Consulta:** {entry['query']}\n"
                result += f"  **Resposta:** {entry['response']}\n"
            result += "\n"
        
        # Adicionar informações gramaticais se disponíveis
        if 'grammar' in categories and categories['grammar']:
            result += "### Gramática\n"
            for entry in categories['grammar'][:3]:  # Mostrar até 3 entradas gramaticais
                result += f"- **Consulta:** {entry['query']}\n"
                result += f"  **Resposta:** {entry['response']}\n"
            result += "\n"
        
        # Adicionar comparações se disponíveis
        if 'comparison' in categories and categories['comparison']:
            result += "### Comparações\n"
            for entry in categories['comparison'][:3]:  # Mostrar até 3 comparações
                result += f"- **Consulta:** {entry['query']}\n"
                result += f"  **Resposta:** {entry['response']}\n"
            result += "\n"
        
        # Adicionar informações de como usar se disponíveis
        if 'how_to' in categories and categories['how_to']:
            result += "### Como Usar\n"
            for entry in categories['how_to'][:3]:  # Mostrar até 3 instruções
                result += f"- **Consulta:** {entry['query']}\n"
                result += f"  **Resposta:** {entry['response']}\n"
            result += "\n"
        
        # Adicionar outras informações se disponíveis
        if 'other' in categories and categories['other']:
            result += "### Outras Informações\n"
            for entry in categories['other'][:3]:  # Mostrar até 3 outras informações
                result += f"- **Consulta:** {entry['query']}\n"
                result += f"  **Resposta:** {entry['response']}\n"
            result += "\n"
        
        # Se não houver informações disponíveis
        if not categories:
            if language == 'yanomami':
                result += "Não foram encontradas informações sobre esta palavra ou frase em Yanomami.\n"
            else:
                result += "Não foram encontradas informações sobre como traduzir esta palavra ou frase para Yanomami.\n"
        
        return result

def main():
    """Função principal para demonstração do tradutor RAG."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Yanomami RAG Translator')
    parser.add_argument('--dataset_dir', type=str, default='yanomami_dataset',
                        help='Diretório contendo os arquivos JSONL do dataset')
    parser.add_argument('--model_dir', type=str, default='gpt2_yanomami_translator',
                        help='Diretório contendo o modelo GPT-2 fine-tuned')
    parser.add_argument('--rebuild_index', action='store_true',
                        help='Reconstruir o índice mesmo se já existir')
    args = parser.parse_args()
    
    # Se rebuild_index for True, remover arquivos de índice existentes
    if args.rebuild_index:
        if os.path.exists('yanomami_index.faiss'):
            os.remove('yanomami_index.faiss')
        if os.path.exists('yanomami_entries.json'):
            os.remove('yanomami_entries.json')
    
    # Inicializar o tradutor
    translator = YanomamiRAGTranslator(
        dataset_dir=args.dataset_dir,
        gpt2_model_dir=args.model_dir
    )
    
    # Interface de linha de comando simples
    print("\n===== Yanomami RAG Translator =====")
    print("Digite 'sair' para encerrar.\n")
    
    while True:
        query = input("\nDigite sua consulta: ")
        if query.lower() in ['sair', 'exit', 'quit']:
            break
            
        try:
            print("\nPesquisando e gerando tradução...")
            result = translator.translate(query)
            print("\nResultado:")
            print(result)
        except Exception as e:
            logger.error(f"Erro ao traduzir: {e}")
            print(f"\nErro ao processar sua consulta: {e}")

if __name__ == "__main__":
    main()
