import pandas as pd
import numpy as np
import torch
from typing import List, Tuple
from transformers import TFBertForTokenClassification, BertTokenizerFast, create_optimizer
import tensorflow as tf

# Проверка доступности GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Установка стратегии распределения для использования всех доступных GPU
        # Если у вас несколько GPU, это распределит обучение между ними
        strategy = tf.distribute.MirroredStrategy()
        print(f"Обнаружено GPU: {len(gpus)}. Используется стратегия MirroredStrategy.")
        # Весь код обучения модели должен быть внутри этого блока
        # (модель, оптимизатор, компиляция и fit)
    except RuntimeError as e:
        print(f"Ошибка при настройке GPU: {e}")
        print("Обучение будет продолжено на CPU.")
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0") # Fallback на CPU
else:
    print("GPU не обнаружен. Обучение будет продолжено на CPU.")
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0") # Явно указываем CPU

# --- Конфигурация ---
TRAIN_FILE = 'data/eng.train'
VAL_FILE = 'data/eng.testa' # Используется как валидационная выборка
TEST_FILE = 'data/eng.testb' # Используется как тестовая выборка

# Определим список всех уникальных тегов сущностей в CoNLL-2003
NER_TAG_NAMES = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
TAG_TO_ID = {tag: i for i, tag in enumerate(NER_TAG_NAMES)}
ID_TO_TAG = {i: tag for tag, i in TAG_TO_ID.items()}
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

# --- Подготовка данных для BERT ---
MAX_SEQ_LENGTH = 128 

def tokenize_and_align_labels_bert(sentences: List[dict], tokenizer: BertTokenizerFast, max_length: int, tag_to_id: dict, id_to_tag: dict):
    """
    Токенизация предложений с использованием токенизатора BERT
    и выравнивание меток NER с токенами BERT (с учетом суб-единиц).
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

# --- Определение и обучение модели BERT ---

def train_bert_model(train_data_raw: List[dict], val_data_raw: List[dict], test_data_raw: List[dict], tag_to_id_map: dict, id_to_tag_map: dict, num_epochs: int = 3, batch_size: int = 16, learning_rate: float = 5e-5):
    """
    Обучение модели BERT для задачи NER.
    """
    print("Обучение модели BERT...")

    with strategy.scope():
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        model = TFBertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(tag_to_id_map))

        # Токенизация и выравнивание меток
        train_encoded = tokenize_and_align_labels_bert(train_data_raw, tokenizer, MAX_SEQ_LENGTH, tag_to_id_map, id_to_tag_map)
        val_encoded = tokenize_and_align_labels_bert(val_data_raw, tokenizer, MAX_SEQ_LENGTH, tag_to_id_map, id_to_tag_map)
        test_encoded = tokenize_and_align_labels_bert(test_data_raw, tokenizer, MAX_SEQ_LENGTH, tag_to_id_map, id_to_tag_map)

        # Создание TensorFlow Datasets
        train_dataset_bert = tf.data.Dataset.from_tensor_slices((
            dict(train_encoded),
            train_encoded["labels"]
        )).batch(batch_size)

        val_dataset_bert = tf.data.Dataset.from_tensor_slices((
            dict(val_encoded),
            val_encoded["labels"]
        )).batch(batch_size)

        test_dataset_bert = tf.data.Dataset.from_tensor_slices((
            dict(test_encoded),
            test_encoded["labels"]
        )).batch(batch_size)

        # Компиляция модели
        optimizer, schedule = create_optimizer(
            init_lr=learning_rate,
            num_warmup_steps=0,
            num_train_steps=len(train_dataset_bert) * num_epochs,
            weight_decay_rate=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-6
        )
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

        # Обучение модели
        model.fit(train_dataset_bert, epochs=num_epochs, validation_data=val_dataset_bert)

    print("Обучение BERT завершено.")
    return model, test_dataset_bert

def evaluate_bert_model(model: TFBertForTokenClassification, test_dataset_bert: tf.data.Dataset, id_to_tag_map: dict):
    """
    Оценка производительности модели BERT.
    """
    print("Оценка модели BERT...")
    predictions = []
    true_labels = []

    from seqeval.metrics import classification_report # Импортируем здесь, чтобы избежать конфликтов

    for batch in test_dataset_bert:
        inputs, labels = batch
        logits = model(inputs).logits
        predicted_ids = tf.argmax(logits, axis=-1).numpy()

        for i in range(len(labels)):
            sentence_labels = labels[i].numpy()
            sentence_predictions = predicted_ids[i]
            attention_mask = inputs['attention_mask'][i].numpy()
            
            current_true_labels = []
            current_predictions = []
            
            for j in range(len(sentence_labels)): # Итерируем по всей длине, чтобы учесть padding
                if attention_mask[j] == 1 and sentence_labels[j] != PAD_TAG_ID: # Пропускаем PAD токены и токены вне маски
                    current_true_labels.append(id_to_tag_map[sentence_labels[j]])
                    current_predictions.append(id_to_tag_map[sentence_predictions[j]])
            
            if current_true_labels: # Убедимся, что список не пуст
                true_labels.append(current_true_labels)
                predictions.append(current_predictions)

    print("\nОтчет по классификации BERT:")
    print(classification_report(true_labels, predictions))
    print("Оценка BERT завершена.")
    return true_labels, predictions

if __name__ == "__main__":
    # 1. Загрузка данных
    train_data_raw, val_data_raw, test_data_raw = load_data(TRAIN_FILE, VAL_FILE, TEST_FILE)

    # Преобразование строковых тегов в ID для всех данных
    def add_tag_ids_to_sentences(sentences: List[dict], tag_to_id_map: dict) -> List[dict]:
        for sentence in sentences:
            sentence['ner_tags_ids'] = [tag_to_id_map.get(tag, tag_to_id_map['O']) for tag in sentence['ner_tags']]
        return sentences

    train_data_processed = add_tag_ids_to_sentences(train_data_raw, TAG_TO_ID)
    val_data_processed = add_tag_ids_to_sentences(val_data_raw, TAG_TO_ID)
    test_data_processed = add_tag_ids_to_sentences(test_data_raw, TAG_TO_ID)

    # Обучение и оценка модели BERT
    NUM_EPOCHS_BERT = 1 # Уменьшено для ускорения обучения
    BATCH_SIZE_BERT = 16 # Меньший размер батча для BERT из-за потребления памяти
    LEARNING_RATE_BERT = 5e-5

    bert_model, bert_test_dataset = train_bert_model(
        train_data_processed, # Используем processed данные
        val_data_processed,   # Используем processed данные
        test_data_processed,  # Используем processed данные
        TAG_TO_ID,
        ID_TO_TAG,
        NUM_EPOCHS_BERT,
        BATCH_SIZE_BERT,
        LEARNING_RATE_BERT
    )

    bert_true_labels, bert_predictions = evaluate_bert_model(bert_model, bert_test_dataset, ID_TO_TAG)

    print("\nРеализация BERT завершена. Выполните скрипт для обучения и оценки.")