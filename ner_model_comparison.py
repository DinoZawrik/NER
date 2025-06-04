import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

# Добавляем путь к site-packages текущего окружения Conda
# Это может быть необходимо, если Python не находит установленные пакеты
conda_env_path = os.path.join(os.path.dirname(sys.executable), '..', 'Lib', 'site-packages')
if conda_env_path not in sys.path:
    sys.path.insert(0, conda_env_path)

# Импорты для CRF и трансформеров будут добавлены позже
# Импорты для CRF и трансформеров будут добавлены позже
# import sklearn_crfsuite
# import nltk
# # nltk.download('punkt')
# # nltk.download('averaged_perceptron_tagger')

# Для CRF слоя будем использовать готовую реализацию
# Убедитесь, что у вас установлена библиотека torchcrf: pip install torchcrf
from TorchCRF import CRF

# def word2features(sent, i):
#     word = sent['tokens'][i]
#     postag = nltk.pos_tag([word])[0][1]

#     features = {
#         'bias': 1.0,
#         'word.lower()': word.lower(),
#         'word[-3:]': word[-3:],
#         'word[-2:]': word[-2:],
#         'word.isupper()': word.isupper(),
#         'word.istitle()': word.istitle(),
#         'word.isdigit()': word.isdigit(),
#         'postag': postag,
#         'postag[:2]': postag[:2],
#     }
#     if i > 0:
#         word1 = sent['tokens'][i-1]
#         postag1 = nltk.pos_tag([word1])[0][1]
#         features.update({
#             '-1:word.lower()': word1.lower(),
#             '-1:word.istitle()': word1.istitle(),
#             '-1:word.isupper()': word1.isupper(),
#             '-1:postag': postag1,
#             '-1:postag[:2]': postag1[:2],
#         })
#     else:
#         features['BOS'] = True

#     if i < len(sent['tokens'])-1:
#         word1 = sent['tokens'][i+1]
#         postag1 = nltk.pos_tag([word1])[0][1]
#         features.update({
#             '+1:word.lower()': word1.lower(),
#             '+1:word.istitle()': word1.istitle(),
#             '+1:word.isupper()': word1.isupper(),
#             '+1:postag': postag1,
#             '+1:postag[:2]': postag1[:2],
#         })
#     else:
#         features['EOS'] = True

#     return features

# def sent2features(sent):
#     return [word2features(sent, i) for i in range(len(sent['tokens']))]

# def sent2labels(sent):
#     return [tag for tag in sent['ner_tags']]

# def sent2tokens(sent):
#     return [token for token in sent['tokens']]

# --- Конфигурация ---
TRAIN_FILE = 'data/eng.train'
VAL_FILE = 'data/eng.testa' # Используется как валидационная выборка
TEST_FILE = 'data/eng.testb' # Используется как тестовая выборка

# Определим список всех уникальных тегов сущностей в CoNLL-2003
# Порядок важен, особенно для тега 'O' (обычно 0)
NER_TAG_NAMES = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
TAG_TO_ID = {tag: i for i, tag in enumerate(NER_TAG_NAMES)}
ID_TO_TAG = {i: tag for tag, i in TAG_TO_ID.items()}
# Добавляем специальные токены для padding и неизвестных слов
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_TAG_ID = TAG_TO_ID['O'] # Используем тег 'O' для padding

# --- Загрузка и парсинг данных ---

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
            if not line: # Пустая строка - конец предложения
                if tokens: # Добавляем предложение, если оно не пустое
                    sentences.append({'tokens': tokens, 'ner_tags': ner_tags})
                tokens = []
                ner_tags = []
            else:
                # Ожидаем формат: word POS chunk NER
                parts = line.split()
                if len(parts) >= 4: # Убедимся, что строка содержит как минимум 4 колонки
                    tokens.append(parts[0])
                    ner_tags.append(parts[3])
                # Можно добавить обработку ошибок или пропустить некорректные строки
    # Добавляем последнее предложение, если файл не заканчивается пустой строкой
    if tokens:
        sentences.append({'tokens': tokens, 'ner_tags': ner_tags})
    return sentences

def load_data(train_file: str, val_file: str, test_file: str) -> Tuple[List[dict], List[dict], List[dict]]:
    """Загружает и парсит данные из файлов CoNLL."""
    print("Загрузка данных...")
    train_data = parse_conll_file(train_file)
    val_data = parse_conll_file(val_file)
    test_data = parse_conll_file(test_file)
    print(f"Загружено предложений: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

# --- Построение словаря и преобразование в ID ---

def build_vocab(sentences: List[dict], min_freq: int = 1) -> Tuple[dict, dict, list]:
    """Строит словарь токенов из списка предложений."""
    token_counts = Counter(token for sentence in sentences for token in sentence['tokens'])
    vocab = [token for token, count in token_counts.items() if count >= min_freq]
    vocab = [PAD_TOKEN, UNK_TOKEN] + sorted(vocab) # Добавляем специальные токены
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for token, i in token_to_id.items()}
    print(f"Построен словарь токенов размером: {len(vocab)}")
    return token_to_id, id_to_token, vocab

def tokens_to_ids(sentences: List[dict], token_to_id_map: dict) -> List[List[int]]:
    """Преобразует токены в числовые ID."""
    token_ids_sentences = []
    for sentence in sentences:
        token_ids = [token_to_id_map.get(token, token_to_id_map[UNK_TOKEN]) for token in sentence['tokens']]
        token_ids_sentences.append(token_ids)
    return token_ids_sentences

def tags_to_ids(sentences: List[dict], tag_to_id_map: dict) -> List[List[int]]:
    """Преобразует строковые теги NER в числовые ID."""
    ner_tags_ids_sentences = []
    for sentence in sentences:
        ner_tags_ids = [tag_to_id_map.get(tag, tag_to_id_map['O']) for tag in sentence['ner_tags']]
        ner_tags_ids_sentences.append(ner_tags_ids)
    return ner_tags_ids_sentences

# --- Подготовка данных для PyTorch (Dataset и DataLoader) ---

class NERDataset(Dataset):
    """Custom Dataset для данных NER."""
    def __init__(self, token_ids: List[List[int]], ner_tags_ids: List[List[int]]):
        self.token_ids = token_ids
        self.ner_tags_ids = ner_tags_ids

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.token_ids[idx], dtype=torch.long), torch.tensor(self.ner_tags_ids[idx], dtype=torch.long)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Функция для объединения батча данных с padding."""
    tokens, tags = zip(*batch)
    # Добавляем padding
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=token_to_id[PAD_TOKEN])
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=PAD_TAG_ID)
    # Создаем маску для игнорирования padding при расчете потерь и в CRF
    attention_mask = (tokens_padded != token_to_id[PAD_TOKEN]).bool()
    return tokens_padded, tags_padded, attention_mask

# --- Определение модели Bi-LSTM-CRF ---

class BiLSTM_CRF(nn.Module):
    """Модель Bi-LSTM-CRF для задачи NER."""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_tags: int, dropout_rate: float = 0.1):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_tags = num_tags

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=token_to_id[PAD_TOKEN])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

        # Линейный слой для проекции выхода LSTM в пространство тегов
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)

        # CRF слой
        self.crf = CRF(num_tags)

    def _get_lstm_features(self, sentence: torch.Tensor) -> torch.Tensor:
        """Получает выходные признаки из Bi-LSTM."""
        embeds = self.embedding(sentence)
        embeds = self.dropout(embeds) # Применяем Dropout к эмбеддингам
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out) # Применяем Dropout к выходу LSTM
        lstm_features = self.hidden2tag(lstm_out)
        return lstm_features

    def forward(self, sentence: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели для расчета функции потерь CRF.
        Args:
            sentence: Тензор с ID токенов (batch_size, seq_len)
            tags: Тензор с ID тегов (batch_size, seq_len)
            mask: Маска для игнорирования padding (batch_size, seq_len)
        Returns:
            Тензор с отрицательной логарифмической вероятностью (loss).
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
            feats: Тензор с логитами из LSTM (batch_size, seq_len, num_tags)
            mask: Булева маска для игнорирования padding (batch_size, seq_len)
        Returns:
            Список списков с предсказанными ID тегов для каждого предложения в батче.
        """
        batch_size, seq_len, num_tags = feats.shape
        decoded_paths = []

        # Переводим тензоры на CPU для обработки
        feats = feats.cpu()
        mask = mask.cpu()

        for i in range(batch_size):
            sentence_feats = feats[i, mask[i]] # Получаем признаки только для реальных токенов
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
            viterbi_scores[0] = sentence_feats[0] + start_transitions # start_transitions - это параметры CRF

            # Прямой проход (Forward Pass)
            for t in range(1, sentence_len):
                # score_t[j] = max score of a path ending at tag j at step t
                # for each tag j, we consider all possible previous tags k
                # score_t[j] = max_k (viterbi_scores[t-1][k] + transitions[k][j] + emission_score[t][j])
                
                # Расширяем viterbi_scores[t-1] для broadcast
                prev_scores = viterbi_scores[t-1].unsqueeze(1) # (num_tags, 1)
                # Расширяем emission_score[t] для broadcast
                emission_score = sentence_feats[t].unsqueeze(0) # (1, num_tags)

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
            path = [0] * sentence_len # Инициализируем путь нулями
            
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
            sentence: Тензор с ID токенов (batch_size, seq_len)
            mask: Маска для игнорирования padding (batch_size, seq_len)
        Returns:
            Список списков с предсказанными ID тегов для каждого предложения в батче.
        """
        lstm_features = self._get_lstm_features(sentence)
        # Используем ручную реализацию декодирования Витерби
        decoded_tags = self._viterbi_decode_manual(lstm_features, mask)
        return decoded_tags


# --- Обучение и оценка моделей ---

def train_lstm_crf_model(train_dataloader: DataLoader, val_dataloader: DataLoader, vocab_size: int, tag_to_id_map: dict, embedding_dim: int = 100, hidden_dim: int = 256, num_epochs: int = 10, learning_rate: float = 0.01):
    """
    Обучение модели Bi-LSTM-CRF.
    """
    print("Обучение модели Bi-LSTM-CRF...")
    num_tags = len(tag_to_id_map)
    model = BiLSTM_CRF(vocab_size, embedding_dim, hidden_dim, num_tags)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for tokens, tags, mask in train_dataloader:
            tokens, tags, mask = tokens.to(device), tags.to(device), mask.to(device)

            model.zero_grad()

            # Прямой проход и расчет потерь (используем forward метод модели с CRF)
            loss = model(tokens, tags, mask).mean() # Усредняем потери по батчу

            # Обратное распространение и оптимизация
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Оценка на валидационной выборке (расчет потерь)
        model.eval()
        val_loss = 0
        with torch.no_grad():
             for tokens, tags, mask in val_dataloader:
                tokens, tags, mask = tokens.to(device), tags.to(device), mask.to(device)
                loss = model(tokens, tags, mask).mean() # Усредняем потери по батчу
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}")

    print("Обучение завершено.")
    return model

def evaluate_model(model: BiLSTM_CRF, dataloader: DataLoader, id_to_tag_map: dict, device: torch.device) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Оценивает производительность модели на данных.
    Использует метрики Precision, Recall, F1-score для NER.
    """
    print("Оценка модели...")
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for tokens, tags, mask in dataloader:
            tokens, tags, mask = tokens.to(device), tags.to(device), mask.to(device)

            # Получаем предсказания с использованием декодирования Витерби
            predicted_tag_ids_batch = model.decode(tokens, mask)

            # Преобразуем ID обратно в строковые метки и удаляем padding
            for i in range(len(predicted_tag_ids_batch)): # Итерация по батчу
                sentence_predictions = [id_to_tag_map[tag_id] for tag_id in predicted_tag_ids_batch[i]]
                # Истинные метки также нужно преобразовать и удалить padding
                sentence_true_labels = [id_to_tag_map[tags[i, j].item()] for j in range(tags.size(1)) if mask[i, j].item() == 1]

                predictions.append(sentence_predictions)
                true_labels.append(sentence_true_labels)

    # Расчет метрик NER (требуется библиотека seqeval)
    # from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
    # print(classification_report(true_labels, predictions))

    print("Оценка завершена (требуется seqeval для полного отчета).")
    # Возвращаем предсказания и истинные метки для дальнейшего анализа
    return true_labels, predictions


# --- Сравнение результатов ---

def compare_results(results):
    """
    Сравнивает результаты оценки различных моделей.
    """
    print("Сравнение результатов (заглушка)...")
    # Здесь будет код для вывода сводной таблицы или графиков сравнения метрик
    # print(pd.DataFrame(results))
    pass

# --- Основная часть скрипта ---

if __name__ == "__main__":
    # 1. Загрузка данных
    train_data_raw, val_data_raw, test_data_raw = load_data(TRAIN_FILE, VAL_FILE, TEST_FILE)

    # 2. Построение словаря и преобразование в ID
    token_to_id, id_to_token, vocab = build_vocab(train_data_raw)
    train_token_ids = tokens_to_ids(train_data_raw, token_to_id)
    val_token_ids = tokens_to_ids(val_data_raw, token_to_id)
    test_token_ids = tokens_to_ids(test_data_raw, token_to_id)

    train_tag_ids = tags_to_ids(train_data_raw, TAG_TO_ID)
    val_tag_ids = tags_to_ids(val_data_raw, TAG_TO_ID)
    test_tag_ids = tags_to_ids(test_data_raw, TAG_TO_ID)

    # 3. Подготовка DataLoader
    BATCH_SIZE = 32
    train_dataset = NERDataset(train_token_ids, train_tag_ids)
    val_dataset = NERDataset(val_token_ids, val_tag_ids)
    test_dataset = NERDataset(test_token_ids, test_tag_ids)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4. Реализация и обучение модели Bi-LSTM-CRF
    # Параметры модели (можно экспериментировать)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    NUM_EPOCHS = 5 # Уменьшено для примера, в ВКР может потребоваться больше
    LEARNING_RATE = 0.005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    lstm_crf_model = train_lstm_crf_model(
        train_dataloader,
        val_dataloader,
        len(vocab),
        TAG_TO_ID,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_EPOCHS,
        LEARNING_RATE
    )

    # 5. Оценка модели
    true_labels, predictions = evaluate_model(lstm_crf_model, test_dataloader, ID_TO_TAG, device)

    # 6. Сравнение результатов (пока только для одной модели)
    # Для полного сравнения потребуется реализовать другие модели и собрать их результаты

    print("\\nРеализация Bi-LSTM-CRF завершена. Выполните скрипт для обучения и оценки.")
    print("Для полной оценки установите библиотеку seqeval: pip install seqeval")