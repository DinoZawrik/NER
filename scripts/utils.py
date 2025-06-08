import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from collections import Counter
import nltk
# nltk.download('punkt') # Закомментировано, так как загрузка должна быть выполнена один раз
# nltk.download('averaged_perceptron_tagger') # Закомментировано, так как загрузка должна быть выполнена один раз

# Добавляем путь к site-packages текущего окружения Conda
# Это может быть необходимо, если Python не находит установленные пакеты
CONDA_ENV_PATH = os.path.join(os.path.dirname(sys.executable), '..', 'Lib', 'site-packages')
if CONDA_ENV_PATH not in sys.path:
    sys.path.insert(0, CONDA_ENV_PATH)

import sklearn_crfsuite
from TorchCRF import CRF
from transformers import BertTokenizerFast, TFBertForTokenClassification
import tensorflow as tf

# --- Конфигурация и словари ---
TRAIN_FILE = 'data/eng.train'
VAL_FILE = 'data/eng.testa'
TEST_FILE = 'data/eng.testb'

NER_TAG_NAMES = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
TAG_TO_ID = {tag: i for i, tag in enumerate(NER_TAG_NAMES)}
ID_TO_TAG = {i: tag for tag, i in TAG_TO_ID.items()}
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_TAG_ID = TAG_TO_ID['O']
MAX_SEQ_LENGTH = 128

# --- Функции для обработки данных CoNLL ---

def parse_conll_file(filepath: str) -> List[dict]:
    """
    Парсит файл в формате CoNLL.

    Возвращает список предложений, каждое из которых - словарь с токенами и тегами NER.
    """
    sentences = []
    tokens = []
    ner_tags = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # Пустая строка - конец предложения
                if tokens:  # Добавляем предложение, если оно не пустое
                    sentences.append({'tokens': tokens, 'ner_tags': ner_tags})
                tokens = []
                ner_tags = []
            else:
                # Ожидаем формат: word POS chunk NER
                parts = line.split()
                if len(parts) >= 4:  # Убедимся, что строка содержит как минимум 4 колонки
                    tokens.append(parts[0])
                    ner_tags.append(parts[3])
    # Добавляем последнее предложение, если файл не заканчивается пустой строкой
    if tokens:
        sentences.append({'tokens': tokens, 'ner_tags': ner_tags})
    return sentences

def load_data(train_file: str, val_file: str, test_file: str) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Загружает и парсит данные из файлов CoNLL.

    Args:
        train_file (str): Путь к файлу с тренировочными данными.
        val_file (str): Путь к файлу с валидационными данными.
        test_file (str): Путь к файлу с тестовыми данными.

    Returns:
        Tuple[List[dict], List[dict], List[dict]]: Кортеж из тренировочных, валидационных и тестовых данных.
    """
    print("Загрузка данных...")
    train_data = parse_conll_file(train_file)
    val_data = parse_conll_file(val_file)
    test_data = parse_conll_file(test_file)
    print(f"Загружено предложений: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

def build_vocab(sentences: List[dict], min_freq: int = 1) -> Tuple[dict, dict, list]:
    """
    Строит словарь токенов из списка предложений.

    Args:
        sentences (List[dict]): Список предложений, каждое из которых - словарь с токенами.
        min_freq (int): Минимальная частота токена для включения в словарь.

    Returns:
        Tuple[dict, dict, list]: Кортеж из словаря токен -> ID, ID -> токен и списка всех токенов в словаре.
    """
    token_counts = Counter(token for sentence in sentences for token in sentence['tokens'])
    vocab = [token for token, count in token_counts.items() if count >= min_freq]
    vocab = [PAD_TOKEN, UNK_TOKEN] + sorted(vocab)  # Добавляем специальные токены
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for token, i in token_to_id.items()}
    print(f"Построен словарь токенов размером: {len(vocab)}")
    return token_to_id, id_to_token, vocab

def tokens_to_ids(sentences: List[dict], token_to_id_map: dict) -> List[List[int]]:
    """
    Преобразует токены в числовые ID.

    Args:
        sentences (List[dict]): Список предложений с токенами.
        token_to_id_map (dict): Словарь для преобразования токенов в ID.

    Returns:
        List[List[int]]: Список списков ID токенов.
    """
    token_ids_sentences = []
    for sentence in sentences:
        token_ids = [token_to_id_map.get(token, token_to_id_map[UNK_TOKEN]) for token in sentence['tokens']]
        token_ids_sentences.append(token_ids)
    return token_ids_sentences

def tags_to_ids(sentences: List[dict], tag_to_id_map: dict) -> List[List[int]]:
    """
    Преобразует строковые теги NER в числовые ID.

    Args:
        sentences (List[dict]): Список предложений с тегами NER.
        tag_to_id_map (dict): Словарь для преобразования тегов в ID.

    Returns:
        List[List[int]]: Список списков ID тегов NER.
    """
    ner_tags_ids_sentences = []
    for sentence in sentences:
        ner_tags_ids = [tag_to_id_map.get(tag, tag_to_id_map['O']) for tag in sentence['ner_tags']]
        ner_tags_ids_sentences.append(ner_tags_ids)
    return ner_tags_ids_sentences

def add_tag_ids_to_sentences(sentences: List[dict], tag_to_id_map: dict) -> List[dict]:
    """
    Добавляет числовые ID тегов NER к каждому предложению.

    Args:
        sentences (List[dict]): Список предложений.
        tag_to_id_map (dict): Словарь для преобразования тегов в ID.

    Returns:
        List[dict]: Список предложений с добавленными ID тегов.
    """
    for sentence in sentences:
        sentence['ner_tags_ids'] = [tag_to_id_map.get(tag, tag_to_id_map['O']) for tag in sentence['ner_tags']]
    return sentences

# --- Функции для CRF модели (sklearn-crfsuite) ---

def word2features(sent: dict, i: int) -> dict:
    """
    Извлекает признаки для слова в предложении для CRF модели.

    Args:
        sent (dict): Словарь, представляющий предложение, с ключами 'tokens'.
        i (int): Индекс слова в предложении.

    Returns:
        dict: Словарь признаков для слова.
    """
    word = sent['tokens'][i]
    # Используем nltk.pos_tag для получения части речи
    # Замечание: nltk.download('punkt') и nltk.download('averaged_perceptron_tagger')
    # должны быть выполнены перед использованием.
    postag = nltk.pos_tag([word])[0][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent['tokens'][i-1]
        postag1 = nltk.pos_tag([word1])[0][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True  # Начало предложения

    if i < len(sent['tokens']) - 1:
        word1 = sent['tokens'][i+1]
        postag1 = nltk.pos_tag([word1])[0][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True  # Конец предложения

    return features

def sent2features(sent: dict) -> List[dict]:
    """
    Извлекает признаки для всех слов в предложении.

    Args:
        sent (dict): Словарь, представляющий предложение.

    Returns:
        List[dict]: Список словарей признаков для каждого слова.
    """
    return [word2features(sent, i) for i in range(len(sent['tokens']))]

def sent2labels(sent: dict) -> List[str]:
    """
    Извлекает метки NER для всех слов в предложении.

    Args:
        sent (dict): Словарь, представляющий предложение.

    Returns:
        List[str]: Список строковых меток NER.
    """
    return [tag for tag in sent['ner_tags']]

def sent2tokens(sent: dict) -> List[str]:
    """
    Извлекает токены для всех слов в предложении.

    Args:
        sent (dict): Словарь, представляющий предложение.

    Returns:
        List[str]: Список токенов.
    """
    return [token for token in sent['tokens']]

# --- Классы и функции для PyTorch (Bi-LSTM-CRF) ---

class NERDataset(Dataset):
    """Custom Dataset для данных NER."""
    def __init__(self, token_ids: List[List[int]], ner_tags_ids: List[List[int]]):
        """
        Инициализирует NERDataset.

        Args:
            token_ids (List[List[int]]): Список списков ID токенов.
            ner_tags_ids (List[List[int]]): Список списков ID тегов NER.
        """
        self.token_ids = token_ids
        self.ner_tags_ids = ner_tags_ids

    def __len__(self) -> int:
        """Возвращает количество предложений в датасете."""
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает тензоры токенов и тегов для заданного индекса.

        Args:
            idx (int): Индекс элемента.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Кортеж из тензора токенов и тензора тегов.
        """
        return torch.tensor(self.token_ids[idx], dtype=torch.long), torch.tensor(self.ner_tags_ids[idx], dtype=torch.long)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_token_id: int, pad_tag_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Функция для объединения батча данных с padding.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): Список кортежей (токены, теги) для батча.
        pad_token_id (int): ID токена для padding.
        pad_tag_id (int): ID тега для padding.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Кортеж из padded токенов, padded тегов и маски внимания.
    """
    tokens, tags = zip(*batch)
    # Добавляем padding
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=pad_token_id)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=pad_tag_id)
    # Создаем маску для игнорирования padding при расчете потерь и в CRF
    attention_mask = (tokens_padded != pad_token_id).bool()
    return tokens_padded, tags_padded, attention_mask

class BiLSTM_CRF(nn.Module):
    """Модель Bi-LSTM-CRF для задачи NER."""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_tags: int, padding_idx: int, dropout_rate: float = 0.1):
        """
        Инициализирует модель Bi-LSTM-CRF.

        Args:
            vocab_size (int): Размер словаря токенов.
            embedding_dim (int): Размерность эмбеддингов.
            hidden_dim (int): Размерность скрытого состояния LSTM.
            num_tags (int): Количество уникальных тегов NER.
            padding_idx (int): Индекс токена padding для nn.Embedding.
            dropout_rate (float): Вероятность dropout.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_tags = num_tags

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

        # Линейный слой для проекции выхода LSTM в пространство тегов
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)

        # CRF слой
        self.crf = CRF(num_tags)

    def _get_lstm_features(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Получает выходные признаки из Bi-LSTM.

        Args:
            sentence (torch.Tensor): Тензор с ID токенов (batch_size, seq_len).

        Returns:
            torch.Tensor: Тензор признаков из LSTM (batch_size, seq_len, hidden_dim).
        """
        embeds = self.embedding(sentence)
        embeds = self.dropout(embeds)  # Применяем Dropout к эмбеддингам
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)  # Применяем Dropout к выходу LSTM
        lstm_features = self.hidden2tag(lstm_out)
        return lstm_features

    def forward(self, sentence: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели для расчета функции потерь CRF.

        Args:
            sentence (torch.Tensor): Тензор с ID токенов (batch_size, seq_len).
            tags (torch.Tensor): Тензор с ID тегов (batch_size, seq_len).
            mask (torch.Tensor): Маска для игнорирования padding (batch_size, seq_len).

        Returns:
            torch.Tensor: Тензор с отрицательной логарифмической вероятностью (loss).
        """
        lstm_features = self._get_lstm_features(sentence)
        # CRF слой принимает логиты (выходы LSTM) и истинные теги для расчета потерь
        # mask=mask указывает CRF слою игнорировать padding
        # reduction='mean' усредняет потери по батчу
        loss = -self.crf(lstm_features, tags, mask=mask)
        return loss

    def _viterbi_decode_manual(self, feats: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """
        Ручная реализация алгоритма Витерби для декодирования последовательности тегов.

        Args:
            feats (torch.Tensor): Тензор с логитами из LSTM (batch_size, seq_len, num_tags).
            mask (torch.Tensor): Булева маска для игнорирования padding (batch_size, seq_len).

        Returns:
            List[List[int]]: Список списков с предсказанными ID тегов для каждого предложения в батче.
        """
        batch_size, seq_len, num_tags = feats.shape
        decoded_paths = []

        # Переводим тензоры на CPU для обработки
        feats = feats.cpu()
        mask = mask.cpu()

        for i in range(batch_size):
            sentence_feats = feats[i, mask[i]]  # Получаем признаки только для реальных токенов
            sentence_len = sentence_feats.shape[0]

            if sentence_len == 0:
                decoded_paths.append([])
                continue

            # Инициализация
            # viterbi_scores[t][j] = max score of a path ending at tag j at step t
            viterbi_scores = torch.full((sentence_len, num_tags), -1e10)
            # backpointers[t][j] = tag that preceded tag j at step t to get max score
            backpointers = torch.full((sentence_len, num_tags), -1, dtype=torch.long)

            # Начальные скоры (для первого токена)
            # Переходы из "START" состояния к первому тегу
            start_transitions = self.crf.state_dict()['start_trans']
            viterbi_scores[0] = sentence_feats[0] + start_transitions  # start_transitions - это параметры CRF

            # Прямой проход (Forward Pass)
            for t in range(1, sentence_len):
                # score_t[j] = max score of a path ending at tag j at step t
                # for each tag j, we consider all possible previous tags k
                # score_t[j] = max_k (viterbi_scores[t-1][k] + transitions[k][j] + emission_score[t][j])
                
                # Расширяем viterbi_scores[t-1] для broadcast
                prev_scores = viterbi_scores[t-1].unsqueeze(1)  # (num_tags, 1)
                # Расширяем emission_score[t] для broadcast
                emission_score = sentence_feats[t].unsqueeze(0)  # (1, num_tags)

                # transitions[k][j] - переход из k в j
                # self.crf.transitions - это матрица (num_tags, num_tags)
                # prev_scores + self.crf.transitions + emission_score
                # (num_tags, 1) + (num_tags, num_tags) + (1, num_tags)
                # = (num_tags, num_tags)
                
                # Вычисляем скоры для всех возможных предыдущих тегов
                transitions = self.crf.state_dict()['trans_matrix']
                scores = prev_scores + transitions + emission_score
                
                # Находим максимальный скор и соответствующий предыдущий тег
                max_scores, max_indices = torch.max(scores, dim=0)
                
                viterbi_scores[t] = max_scores
                backpointers[t] = max_indices

            # Обратный проход (Backward Pass) - восстановление пути
            path = [0] * sentence_len  # Инициализируем путь нулями
            
            # Последний тег: находим тег с максимальным скором на последнем шаге
            # Добавляем переходы к "END" состоянию
            end_transitions = self.crf.state_dict()['end_trans']
            final_scores = viterbi_scores[sentence_len - 1] + end_transitions
            best_last_tag_id = torch.argmax(final_scores).item()
            path[sentence_len - 1] = best_last_tag_id

            # Восстанавливаем путь, используя backpointers
            for t in range(sentence_len - 2, -1, -1):
                path[t] = backpointers[t + 1, path[t + 1]].item()
            
            decoded_paths.append(path)
        return decoded_paths

    def decode(self, sentence: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """
        Декодирование последовательности тегов с использованием алгоритма Витерби.

        Args:
            sentence (torch.Tensor): Тензор с ID токенов (batch_size, seq_len).
            mask (torch.Tensor): Маска для игнорирования padding (batch_size, seq_len).

        Returns:
            List[List[int]]: Список списков с предсказанными ID тегов для каждого предложения в батче.
        """
        lstm_features = self._get_lstm_features(sentence)
        # Используем ручную реализацию декодирования Витерби
        decoded_tags = self._viterbi_decode_manual(lstm_features, mask)
        return decoded_tags

# --- Функции для BERT модели ---

def tokenize_and_align_labels_bert(sentences: List[dict], tokenizer: BertTokenizerFast, max_length: int, tag_to_id: dict, id_to_tag: dict):
    """
    Токенизация предложений с использованием токенизатора BERT
    и выравнивание меток NER с токенами BERT (с учетом суб-единиц).

    Args:
        sentences (List[dict]): Список предложений, каждое из которых - словарь с токенами и ID тегов.
        tokenizer (BertTokenizerFast): Токенизатор BERT.
        max_length (int): Максимальная длина последовательности.
        tag_to_id (dict): Словарь для преобразования строковых тегов в ID.
        id_to_tag (dict): Словарь для преобразования ID в строковые теги.

    Returns:
        transformers.tokenization_utils_base.BatchEncoding: Токенизированные входные данные с метками.
    """
    tokenized_inputs = tokenizer(
        [s['tokens'] for s in sentences],
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=False,
        return_offsets_mapping=False,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="tf"
    )

    labels = []
    for i, sentence in enumerate(sentences):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(PAD_TAG_ID)
            elif word_idx != previous_word_idx:
                label_ids.append(sentence['ner_tags_ids'][word_idx])
            else:
                tag_id = sentence['ner_tags_ids'][word_idx]
                if id_to_tag[tag_id].startswith('B-'):
                    i_tag = 'I-' + id_to_tag[tag_id][2:]
                    label_ids.append(tag_to_id[i_tag])
                else:
                    label_ids.append(tag_id)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = tf.constant(labels, dtype=tf.int64)
    return tokenized_inputs