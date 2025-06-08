import tensorflow as tf
from transformers import TFBertForTokenClassification, BertTokenizerFast, create_optimizer
from typing import List, Tuple
from seqeval.metrics import classification_report

# Импортируем общие утилиты и классы
from scripts.utils import (
    TRAIN_FILE, VAL_FILE, TEST_FILE,
    NER_TAG_NAMES, TAG_TO_ID, ID_TO_TAG, PAD_TAG_ID, MAX_SEQ_LENGTH,
    load_data, add_tag_ids_to_sentences, tokenize_and_align_labels_bert
)

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
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")  # Fallback на CPU
else:
    print("GPU не обнаружен. Обучение будет продолжено на CPU.")
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")  # Явно указываем CPU

def train_bert_model(
    train_data_raw: List[dict],
    val_data_raw: List[dict],
    tag_to_id_map: dict,
    id_to_tag_map: dict,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5
) -> Tuple[TFBertForTokenClassification, tf.data.Dataset]:
    """
    Обучает модель BERT для задачи NER.

    Args:
        train_data_raw (List[dict]): Список тренировочных предложений.
        val_data_raw (List[dict]): Список валидационных предложений.
        tag_to_id_map (dict): Словарь для преобразования строковых тегов в ID.
        id_to_tag_map (dict): Словарь для преобразования ID в строковые теги.
        num_epochs (int): Количество эпох обучения.
        batch_size (int): Размер батча.
        learning_rate (float): Скорость обучения.

    Returns:
        Tuple[TFBertForTokenClassification, tf.data.Dataset]:
            Кортеж из обученной модели BERT и тестового датасета BERT.
    """
    print("Обучение модели BERT...")

    with strategy.scope():
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        model = TFBertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(tag_to_id_map))

        # Токенизация и выравнивание меток
        train_encoded = tokenize_and_align_labels_bert(
            train_data_raw, tokenizer, MAX_SEQ_LENGTH, tag_to_id_map, id_to_tag_map
        )
        val_encoded = tokenize_and_align_labels_bert(
            val_data_raw, tokenizer, MAX_SEQ_LENGTH, tag_to_id_map, id_to_tag_map
        )

        # Создание TensorFlow Datasets
        train_dataset_bert = tf.data.Dataset.from_tensor_slices((
            dict(train_encoded),
            train_encoded["labels"]
        )).batch(batch_size)

        val_dataset_bert = tf.data.Dataset.from_tensor_slices((
            dict(val_encoded),
            val_encoded["labels"]
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
    return model

def evaluate_bert_model(
    model: TFBertForTokenClassification,
    test_dataset_bert: tf.data.Dataset,
    id_to_tag_map: dict
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Оценивает производительность модели BERT.

    Args:
        model (TFBertForTokenClassification): Обученная модель BERT.
        test_dataset_bert (tf.data.Dataset): Тестовый датасет BERT.
        id_to_tag_map (dict): Словарь для преобразования ID тегов в строковые теги.

    Returns:
        Tuple[List[List[str]], List[List[str]]]: Кортеж из истинных меток и предсказанных меток.
    """
    print("Оценка модели BERT...")
    predictions = []
    true_labels = []

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
            
            for j in range(len(sentence_labels)):  # Итерируем по всей длине, чтобы учесть padding
                if attention_mask[j] == 1 and sentence_labels[j] != PAD_TAG_ID:  # Пропускаем PAD токены и токены вне маски
                    current_true_labels.append(id_to_tag_map[sentence_labels[j]])
                    current_predictions.append(id_to_tag_map[sentence_predictions[j]])
            
            if current_true_labels:  # Убедимся, что список не пуст
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
    train_data_processed = add_tag_ids_to_sentences(train_data_raw, TAG_TO_ID)
    val_data_processed = add_tag_ids_to_sentences(val_data_raw, TAG_TO_ID)
    test_data_processed = add_tag_ids_to_sentences(test_data_raw, TAG_TO_ID)

    # Обучение и оценка модели BERT
    NUM_EPOCHS_BERT = 1  # Уменьшено для ускорения обучения
    BATCH_SIZE_BERT = 16  # Меньший размер батча для BERT из-за потребления памяти
    LEARNING_RATE_BERT = 5e-5

    bert_model = train_bert_model(
        train_data_processed,
        val_data_processed,
        TAG_TO_ID,
        ID_TO_TAG,
        NUM_EPOCHS_BERT,
        BATCH_SIZE_BERT,
        LEARNING_RATE_BERT
    )

    # Подготовка тестового датасета для оценки
    tokenizer_eval = BertTokenizerFast.from_pretrained('bert-base-cased')
    test_encoded_eval = tokenize_and_align_labels_bert(
        test_data_processed, tokenizer_eval, MAX_SEQ_LENGTH, TAG_TO_ID, ID_TO_TAG
    )
    bert_test_dataset_eval = tf.data.Dataset.from_tensor_slices((
        dict(test_encoded_eval),
        test_encoded_eval["labels"]
    )).batch(BATCH_SIZE_BERT)

    bert_true_labels, bert_predictions = evaluate_bert_model(
        bert_model, bert_test_dataset_eval, ID_TO_TAG
    )

    print("\nРеализация BERT завершена. Выполните скрипт для обучения и оценки.")