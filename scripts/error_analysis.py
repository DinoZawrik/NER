import torch
from torch.utils.data import DataLoader
import tensorflow as tf
import joblib
from transformers import BertTokenizerFast, TFBertForTokenClassification
from typing import List, Tuple

# Импортируем общие утилиты и классы
from scripts.utils import (
    TRAIN_FILE, TEST_FILE,
    NER_TAG_NAMES, TAG_TO_ID, ID_TO_TAG, PAD_TOKEN, UNK_TOKEN, PAD_TAG_ID, MAX_SEQ_LENGTH,
    parse_conll_file, build_vocab, tokens_to_ids, tags_to_ids, add_tag_ids_to_sentences,
    NERDataset, collate_fn, BiLSTM_CRF,
    sent2features, sent2labels,
    tokenize_and_align_labels_bert,
)

# --- Основная часть скрипта ---

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # 1. Загрузка данных
    print("Загрузка тестовых данных...")
    test_data_raw = parse_conll_file(TEST_FILE)
    print(f"Загружено предложений в тестовой выборке: {len(test_data_raw)}")

    # Для Bi-LSTM-CRF: нужно загрузить train_data_raw для построения словаря
    print("Загрузка тренировочных данных для построения словаря Bi-LSTM-CRF...")
    train_data_raw = parse_conll_file(TRAIN_FILE)
    print(f"Загружено предложений в тренировочной выборке: {len(train_data_raw)}")

    # 2. Определение словарей (уже определены выше в utils.py)

    # 3. Подготовка данных для Bi-LSTM-CRF
    print("Подготовка данных для Bi-LSTM-CRF...")
    token_to_id, id_to_token, vocab = build_vocab(train_data_raw)
    test_token_ids = tokens_to_ids(test_data_raw, token_to_id)
    test_tag_ids = tags_to_ids(test_data_raw, TAG_TO_ID)

    test_dataset_lstm = NERDataset(test_token_ids, test_tag_ids)
    BATCH_SIZE_LSTM = 32  # Используем тот же размер батча, что и при обучении
    test_dataloader_lstm = DataLoader(
        test_dataset_lstm,
        batch_size=BATCH_SIZE_LSTM,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, token_to_id[PAD_TOKEN], PAD_TAG_ID)
    )
    print("Данные для Bi-LSTM-CRF подготовлены.")

    # 4. Подготовка данных для BERT
    print("Подготовка данных для BERT...")
    tokenizer_bert = BertTokenizerFast.from_pretrained('bert-base-cased')
    test_data_processed_bert = add_tag_ids_to_sentences(test_data_raw, TAG_TO_ID)
    test_encoded_bert = tokenize_and_align_labels_bert(
        test_data_processed_bert, tokenizer_bert, MAX_SEQ_LENGTH, TAG_TO_ID, ID_TO_TAG
    )
    
    BATCH_SIZE_BERT = 16  # Используем тот же размер батча, что и при обучении
    test_dataset_bert = tf.data.Dataset.from_tensor_slices((
        dict(test_encoded_bert),
        test_encoded_bert["labels"]
    )).batch(BATCH_SIZE_BERT)
    print("Данные для BERT подготовлены.")

    # 5. Загрузка моделей
    print("Загрузка моделей...")

    # Загрузка Bi-LSTM-CRF
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    NUM_TAGS = len(NER_TAG_NAMES)
    
    lstm_crf_model = BiLSTM_CRF(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_TAGS, padding_idx=TAG_TO_ID[PAD_TOKEN])
    lstm_crf_model.load_state_dict(torch.load('bilstm_crf_model.pth', map_location=device))
    lstm_crf_model.to(device)
    lstm_crf_model.eval()
    print("Модель Bi-LSTM-CRF загружена.")

    # Загрузка CRF
    crf_model = joblib.load('crf_model.pkl')
    print("Модель CRF загружена.")

    # Инициализация BERT
    bert_model = TFBertForTokenClassification.from_pretrained('bert-base-cased', num_labels=NUM_TAGS)
    # Для BERT, если модель была обучена и сохранена, ее нужно загрузить.
    # В данном случае, если модель не была сохранена, она будет инициализирована с предобученными весами.
    # Если у вас есть сохраненная модель BERT, используйте:
    # bert_model = TFBertForTokenClassification.from_pretrained('./path_to_saved_bert_model', num_labels=NUM_TAGS)
    print("Модель BERT инициализирована (или загружена, если сохранена).")

    # 6. Получение предсказаний
    print("Получение предсказаний...")

    # Предсказания Bi-LSTM-CRF
    lstm_crf_predictions_ids = []
    lstm_crf_true_labels_ids = []
    with torch.no_grad():
        for tokens, tags, mask in test_dataloader_lstm:
            tokens, tags, mask = tokens.to(device), tags.to(device), mask.to(device)
            predicted_tag_ids_batch = lstm_crf_model.decode(tokens, mask)
            lstm_crf_predictions_ids.extend(predicted_tag_ids_batch)
            
            for i in range(len(tags)):
                sentence_true_labels = [
                    tags[i, j].item()
                    for j in range(tags.size(1))
                    if mask[i, j].item() == 1
                ]
                lstm_crf_true_labels_ids.append(sentence_true_labels)

    # Предсказания CRF
    crf_test_features = [sent2features(s) for s in test_data_raw]
    crf_predictions_str = crf_model.predict(crf_test_features)
    crf_true_labels_str = [sent2labels(s) for s in test_data_raw]

    # Предсказания BERT
    bert_predictions_ids = []
    bert_true_labels_ids = []
    for batch in test_dataset_bert:
        inputs, labels = batch
        logits = bert_model(inputs).logits
        predicted_ids = tf.argmax(logits, axis=-1).numpy()

        for i in range(len(labels)):
            sentence_labels = labels[i].numpy()
            sentence_predictions = predicted_ids[i]
            attention_mask = inputs['attention_mask'][i].numpy()
            
            current_true_labels = []
            current_predictions = []
            
            for j in range(len(sentence_labels)):
                if attention_mask[j] == 1 and sentence_labels[j] != PAD_TAG_ID:
                    current_true_labels.append(sentence_labels[j])
                    current_predictions.append(sentence_predictions[j])
            
            if current_true_labels:
                bert_true_labels_ids.append(current_true_labels)
                bert_predictions_ids.append(current_predictions)
    print("Предсказания получены для всех моделей.")

    # 7. Сравнение и вывод ошибок
    print("\n--- Анализ ошибок ---")

    # Преобразуем ID в строковые теги для Bi-LSTM-CRF и BERT
    lstm_crf_predictions_str = [[ID_TO_TAG[tag_id] for tag_id in sent_pred] for sent_pred in lstm_crf_predictions_ids]
    lstm_crf_true_labels_str = [[ID_TO_TAG[tag_id] for tag_id in sent_true] for sent_true in lstm_crf_true_labels_ids]

    bert_predictions_str = [[ID_TO_TAG[tag_id] for tag_id in sent_pred] for sent_pred in bert_predictions_ids]
    bert_true_labels_str = [[ID_TO_TAG[tag_id] for tag_id in sent_true] for sent_true in bert_true_labels_ids]

    # Убедимся, что все списки предсказаний и истинных меток имеют одинаковую длину
    min_len = min(len(test_data_raw), len(lstm_crf_predictions_str), len(crf_predictions_str), len(bert_predictions_str))
    
    # Обрезаем списки до минимальной длины, чтобы избежать ошибок индексации
    test_data_raw = test_data_raw[:min_len]
    lstm_crf_predictions_str = lstm_crf_predictions_str[:min_len]
    lstm_crf_true_labels_str = lstm_crf_true_labels_str[:min_len]
    crf_predictions_str = crf_predictions_str[:min_len]
    crf_true_labels_str = crf_true_labels_str[:min_len]
    bert_predictions_str = bert_predictions_str[:min_len]
    bert_true_labels_str = bert_true_labels_str[:min_len]


    for i in range(min_len):
        sentence_tokens = test_data_raw[i]['tokens']
        true_tags = test_data_raw[i]['ner_tags']  # Истинные теги из исходных данных

        # Проверка длины предсказаний и истинных меток
        # Это важно, так как токенизация BERT может изменить количество токенов
        # Для корректного сравнения, мы должны сравнивать только те токены, которые не являются PAD
        # и которые соответствуют оригинальным словам.
        # Для Bi-LSTM-CRF и CRF, длины должны совпадать с оригинальным предложением.
        # Для BERT, мы уже отфильтровали PAD токены при получении предсказаний.

        # Bi-LSTM-CRF
        lstm_crf_pred = lstm_crf_predictions_str[i]
        # Убедимся, что длины совпадают для сравнения
        if len(lstm_crf_pred) != len(true_tags):
            print(f"Предупреждение: Длина предсказаний Bi-LSTM-CRF ({len(lstm_crf_pred)}) не совпадает с длиной истинных меток ({len(true_tags)}) для предложения: {' '.join(sentence_tokens)}")
            # Пропускаем это предложение или обрабатываем по-другому
            continue

        # CRF
        crf_pred = crf_predictions_str[i]
        if len(crf_pred) != len(true_tags):
            print(f"Предупреждение: Длина предсказаний CRF ({len(crf_pred)}) не совпадает с длиной истинных меток ({len(true_tags)}) для предложения: {' '.join(sentence_tokens)}")
            continue

        # BERT
        bert_pred = bert_predictions_str[i]
        # Для BERT, true_labels_str уже выровнены и отфильтрованы от PAD.
        # Поэтому сравниваем bert_pred с bert_true_labels_str
        bert_true_aligned = bert_true_labels_str[i]
        if len(bert_pred) != len(bert_true_aligned):
            print(f"Предупреждение: Длина предсказаний BERT ({len(bert_pred)}) не совпадает с длиной выровненных истинных меток ({len(bert_true_aligned)}) для предложения: {' '.join(sentence_tokens)}")
            continue
        
        # Сравнение и вывод ошибок
        has_error = False
        error_details = []

        # Bi-LSTM-CRF
        if lstm_crf_pred != true_tags:
            has_error = True
            error_details.append({
                "model": "Bi-LSTM-CRF",
                "true": true_tags,
                "predicted": lstm_crf_pred
            })
        
        # CRF
        if crf_pred != true_tags:
            has_error = True
            error_details.append({
                "model": "CRF",
                "true": true_tags,
                "predicted": crf_pred
            })

        # BERT (сравниваем с выровненными истинными метками)
        # Для BERT, tokens и true_tags должны быть выровнены с токенизацией BERT
        # Это сложнее, так как BERT может разбивать слова на субтокены.
        # Для простоты, будем сравнивать только теги, которые соответствуют оригинальным словам.
        # Если BERT предсказания не совпадают с выровненными истинными метками
        if bert_pred != bert_true_aligned:
            has_error = True
            error_details.append({
                "model": "BERT",
                "true": bert_true_aligned,
                "predicted": bert_pred
            })

        if has_error:
            print("\n" + "="*50)
            print(f"Предложение: {' '.join(sentence_tokens)}")
            print(f"Истинные метки: {true_tags}")
            for error in error_details:
                print(f"  Модель: {error['model']}")
                print(f"    Предсказанные метки: {error['predicted']}")
            print("="*50)

    print("\nАнализ ошибок завершен.")