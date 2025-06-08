import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple
from seqeval.metrics import classification_report, f1_score
import joblib  # Для сохранения CRF модели

# Импортируем общие утилиты и классы
from scripts.utils import (
    TRAIN_FILE, VAL_FILE, TEST_FILE,
    NER_TAG_NAMES, TAG_TO_ID, ID_TO_TAG, PAD_TOKEN, UNK_TOKEN, PAD_TAG_ID,
    parse_conll_file, load_data, build_vocab, tokens_to_ids, tags_to_ids,
    NERDataset, collate_fn, BiLSTM_CRF,
    word2features, sent2features, sent2labels, sent2tokens
)

import sklearn_crfsuite

# --- Обучение и оценка моделей ---

def train_lstm_crf_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    vocab_size: int,
    tag_to_id_map: dict,
    embedding_dim: int = 100,
    hidden_dim: int = 256,
    num_epochs: int = 10,
    learning_rate: float = 0.01
) -> BiLSTM_CRF:
    """
    Обучает модель Bi-LSTM-CRF.

    Args:
        train_dataloader (DataLoader): DataLoader для тренировочных данных.
        val_dataloader (DataLoader): DataLoader для валидационных данных.
        vocab_size (int): Размер словаря токенов.
        tag_to_id_map (dict): Словарь для преобразования строковых тегов в ID.
        embedding_dim (int): Размерность эмбеддингов.
        hidden_dim (int): Размерность скрытого состояния LSTM.
        num_epochs (int): Количество эпох обучения.
        learning_rate (float): Скорость обучения.

    Returns:
        BiLSTM_CRF: Обученная модель Bi-LSTM-CRF.
    """
    print("Обучение модели Bi-LSTM-CRF...")
    num_tags = len(tag_to_id_map)
    model = BiLSTM_CRF(vocab_size, embedding_dim, hidden_dim, num_tags, padding_idx=TAG_TO_ID[PAD_TOKEN])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for tokens, tags, mask in train_dataloader:
            tokens, tags, mask = tokens.to(device), tags.to(device), mask.to(device)

            optimizer.zero_grad()

            # Прямой проход и расчет потерь (используем forward метод модели с CRF)
            loss = model(tokens, tags, mask).mean()  # Усредняем потери по батчу

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
                loss = model(tokens, tags, mask).mean()  # Усредняем потери по батчу
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}")

    print("Обучение завершено.")
    return model

def evaluate_model(
    model: BiLSTM_CRF,
    dataloader: DataLoader,
    id_to_tag_map: dict,
    device: torch.device
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Оценивает производительность модели Bi-LSTM-CRF на данных.

    Использует метрики Precision, Recall, F1-score для NER.

    Args:
        model (BiLSTM_CRF): Обученная модель Bi-LSTM-CRF.
        dataloader (DataLoader): DataLoader для данных оценки.
        id_to_tag_map (dict): Словарь для преобразования ID тегов в строковые теги.
        device (torch.device): Устройство для выполнения вычислений (CPU или GPU).

    Returns:
        Tuple[List[List[str]], List[List[str]]]: Кортеж из истинных меток и предсказанных меток.
    """
    print("Оценка модели Bi-LSTM-CRF...")
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for tokens, tags, mask in dataloader:
            tokens, tags, mask = tokens.to(device), tags.to(device), mask.to(device)

            # Получаем предсказания с использованием декодирования Витерби
            predicted_tag_ids_batch = model.decode(tokens, mask)

            # Преобразуем ID обратно в строковые метки и удаляем padding
            for i in range(len(predicted_tag_ids_batch)):  # Итерация по батчу
                sentence_predictions = [id_to_tag_map[tag_id] for tag_id in predicted_tag_ids_batch[i]]
                # Истинные метки также нужно преобразовать и удалить padding
                sentence_true_labels = [
                    id_to_tag_map[tags[i, j].item()]
                    for j in range(tags.size(1))
                    if mask[i, j].item() == 1
                ]

                predictions.append(sentence_predictions)
                true_labels.append(sentence_true_labels)

    # Расчет метрик NER
    print("\nОтчет по классификации Bi-LSTM-CRF:")
    print(classification_report(true_labels, predictions))

    print("Оценка завершена.")
    # Возвращаем предсказания и истинные метки для дальнейшего анализа
    return true_labels, predictions

def train_crf_model(
    train_data: List[dict],
    test_data: List[dict],
    tag_to_id_map: dict
) -> Tuple[sklearn_crfsuite.CRF, List[List[str]], List[List[str]]]:
    """
    Обучает и оценивает классическую CRF модель.

    Args:
        train_data (List[dict]): Список тренировочных предложений.
        test_data (List[dict]): Список тестовых предложений.
        tag_to_id_map (dict): Словарь для преобразования строковых тегов в ID.

    Returns:
        Tuple[sklearn_crfsuite.CRF, List[List[str]], List[List[str]]]:
            Кортеж из обученной CRF модели, истинных меток и предсказанных меток.
    """
    print("Обучение классической CRF модели...")
    X_train = [sent2features(s) for s in train_data]
    y_train = [sent2labels(s) for s in train_data]

    X_test = [sent2features(s) for s in test_data]
    y_test = [sent2labels(s) for s in test_data]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    print("Обучение CRF завершено.")

    print("\nОценка CRF модели...")
    y_pred = crf.predict(X_test)

    print("\nОтчет по классификации CRF:")
    print(classification_report(y_test, y_pred))
    print("Оценка CRF завершена.")
    return crf, y_test, y_pred

# --- Сравнение результатов ---

def compare_results(results: dict):
    """
    Сравнивает результаты оценки различных моделей и выводит сводную таблицу.

    Args:
        results (dict): Словарь с результатами F1-score для каждой модели.
    """
    print("Сравнение результатов...")
    print(pd.DataFrame.from_dict(results, orient='index', columns=['F1-score']))

# --- Основная часть скрипта ---

if __name__ == "__main__":
    # 1. Загрузка данных
    train_data_raw, val_data_raw, test_data_raw = load_data(TRAIN_FILE, VAL_FILE, TEST_FILE)

    # 2. Построение словаря и преобразование в ID
    token_to_id, id_to_token, vocab = build_vocab(train_data_raw)
    
    # Для Bi-LSTM-CRF
    train_token_ids = tokens_to_ids(train_data_raw, token_to_id)
    val_token_ids = tokens_to_ids(val_data_raw, token_to_id)
    test_token_ids = tokens_to_ids(test_data_raw, token_to_id)

    train_tag_ids = tags_to_ids(train_data_raw, TAG_TO_ID)
    val_tag_ids = tags_to_ids(val_data_raw, TAG_TO_ID)
    test_tag_ids = tags_to_ids(test_data_raw, TAG_TO_ID)

    # 3. Подготовка DataLoader для Bi-LSTM-CRF
    BATCH_SIZE_LSTM = 32
    train_dataset_lstm = NERDataset(train_token_ids, train_tag_ids)
    val_dataset_lstm = NERDataset(val_token_ids, val_tag_ids)
    test_dataset_lstm = NERDataset(test_token_ids, test_tag_ids)

    train_dataloader_lstm = DataLoader(
        train_dataset_lstm,
        batch_size=BATCH_SIZE_LSTM,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, token_to_id[PAD_TOKEN], PAD_TAG_ID)
    )
    val_dataloader_lstm = DataLoader(
        val_dataset_lstm,
        batch_size=BATCH_SIZE_LSTM,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, token_to_id[PAD_TOKEN], PAD_TAG_ID)
    )
    test_dataloader_lstm = DataLoader(
        test_dataset_lstm,
        batch_size=BATCH_SIZE_LSTM,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, token_to_id[PAD_TOKEN], PAD_TAG_ID)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # 4. Реализация и обучение модели Bi-LSTM-CRF
    # Параметры модели (можно экспериментировать)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    NUM_EPOCHS_LSTM = 5  # Уменьшено для примера, в ВКР может потребоваться больше
    LEARNING_RATE_LSTM = 0.005

    lstm_crf_model = train_lstm_crf_model(
        train_dataloader_lstm,
        val_dataloader_lstm,
        len(vocab),
        TAG_TO_ID,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_EPOCHS_LSTM,
        LEARNING_RATE_LSTM
    )

    # Сохранение модели Bi-LSTM-CRF
    torch.save(lstm_crf_model.state_dict(), 'bilstm_crf_model.pth')
    print("Модель Bi-LSTM-CRF сохранена в bilstm_crf_model.pth")

    # 5. Оценка модели Bi-LSTM-CRF
    print("\n--- Оценка Bi-LSTM-CRF ---")
    lstm_crf_true_labels, lstm_crf_predictions = evaluate_model(
        lstm_crf_model, test_dataloader_lstm, ID_TO_TAG, device
    )

    # 6. Реализация и обучение классической CRF модели
    print("\n--- Обучение и оценка CRF ---")
    crf_model, crf_true_labels, crf_predictions = train_crf_model(
        train_data_raw, test_data_raw, TAG_TO_ID
    )

    # Сохранение CRF модели
    joblib.dump(crf_model, 'crf_model.pkl')
    print("Модель CRF сохранена в crf_model.pkl")

    # 9. Сравнение результатов
    print("\n--- Сводка результатов ---")
    results_summary = {
        "Bi-LSTM-CRF": f1_score(lstm_crf_true_labels, lstm_crf_predictions, average='weighted'),
        "CRF": f1_score(crf_true_labels, crf_predictions, average='weighted')
    }
    compare_results(results_summary)

    print("\nРеализация всех моделей завершена. Выполните скрипт для обучения и оценки.")