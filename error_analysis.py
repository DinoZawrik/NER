import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
import joblib
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForTokenClassification
from collections import Counter

# Добавляем путь к site-packages текущего окружения Conda
conda_env_path = os.path.join(os.path.dirname(sys.executable), '..', 'Lib', 'site-packages')
if conda_env_path not in sys.path:
    sys.path.insert(0, conda_env_path)

import sklearn_crfsuite
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from TorchCRF import CRF

# --- Конфигурация и словари из ner_model_comparison.py ---
TRAIN_FILE = 'data/eng.train'
TEST_FILE = 'data/eng.testb'

NER_TAG_NAMES = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
TAG_TO_ID = {tag: i for i, tag in enumerate(NER_TAG_NAMES)}
ID_TO_TAG = {i: tag for tag, i in TAG_TO_ID.items()}
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_TAG_ID = TAG_TO_ID['O']

# --- Функции из ner_model_comparison.py ---

def parse_conll_file(filepath: str) -> List[dict]:
    sentences = []
    tokens = []
    ner_tags = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({'tokens': tokens, 'ner_tags': ner_tags})
                tokens = []
                ner_tags = []
            else:
                parts = line.split()
                if len(parts) >= 4:
                    tokens.append(parts[0])
                    ner_tags.append(parts[3])
    if tokens:
        sentences.append({'tokens': tokens, 'ner_tags': ner_tags})
    return sentences

def build_vocab(sentences: List[dict], min_freq: int = 1) -> Tuple[dict, dict, list]:
    token_counts = Counter(token for sentence in sentences for token in sentence['tokens'])
    vocab = [token for token, count in token_counts.items() if count >= min_freq]
    vocab = [PAD_TOKEN, UNK_TOKEN] + sorted(vocab)
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for token, i in token_to_id.items()}
    return token_to_id, id_to_token, vocab

def tokens_to_ids(sentences: List[dict], token_to_id_map: dict) -> List[List[int]]:
    token_ids_sentences = []
    for sentence in sentences:
        token_ids = [token_to_id_map.get(token, token_to_id_map[UNK_TOKEN]) for token in sentence['tokens']]
        token_ids_sentences.append(token_ids)
    return token_ids_sentences

def tags_to_ids(sentences: List[dict], tag_to_id_map: dict) -> List[List[int]]:
    ner_tags_ids_sentences = []
    for sentence in sentences:
        ner_tags_ids = [tag_to_id_map.get(tag, tag_to_id_map['O']) for tag in sentence['ner_tags']]
        ner_tags_ids_sentences.append(ner_tags_ids)
    return ner_tags_ids_sentences

class NERDataset(Dataset):
    def __init__(self, token_ids: List[List[int]], ner_tags_ids: List[List[int]]):
        self.token_ids = token_ids
        self.ner_tags_ids = ner_tags_ids

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.token_ids[idx], dtype=torch.long), torch.tensor(self.ner_tags_ids[idx], dtype=torch.long)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, tags = zip(*batch)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=token_to_id[PAD_TOKEN])
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=PAD_TAG_ID)
    attention_mask = (tokens_padded != token_to_id[PAD_TOKEN]).bool()
    return tokens_padded, tags_padded, attention_mask

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_tags: int, dropout_rate: float = 0.1):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_tags = num_tags

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=token_to_id[PAD_TOKEN])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags)

    def _get_lstm_features(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_features = self.hidden2tag(lstm_out)
        return lstm_features

    def _viterbi_decode_manual(self, feats: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        batch_size, seq_len, num_tags = feats.shape
        decoded_paths = []

        feats = feats.cpu()
        mask = mask.cpu()

        for i in range(batch_size):
            sentence_feats = feats[i, mask[i]]
            sentence_len = sentence_feats.shape[0]

            if sentence_len == 0:
                decoded_paths.append([])
                continue

            viterbi_scores = torch.full((sentence_len, num_tags), -1e10)
            backpointers = torch.full((sentence_len, num_tags), -1, dtype=torch.long)

            start_transitions = self.crf.state_dict()['start_trans']
            viterbi_scores[0] = sentence_feats[0] + start_transitions

            for t in range(1, sentence_len):
                prev_scores = viterbi_scores[t-1].unsqueeze(1)
                emission_score = sentence_feats[t].unsqueeze(0)
                transitions = self.crf.state_dict()['trans_matrix']
                scores = prev_scores + transitions + emission_score
                
                max_scores, max_indices = torch.max(scores, dim=0)
                
                viterbi_scores[t] = max_scores
                backpointers[t] = max_indices

            path = [0] * sentence_len
            
            end_transitions = self.crf.state_dict()['end_trans']
            final_scores = viterbi_scores[sentence_len - 1] + end_transitions
            best_last_tag_id = torch.argmax(final_scores).item()
            path[sentence_len - 1] = best_last_tag_id

            for t in range(sentence_len - 2, -1, -1):
                path[t] = backpointers[t + 1, path[t + 1]].item()
            
            decoded_paths.append(path)
        return decoded_paths

    def decode(self, sentence: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        lstm_features = self._get_lstm_features(sentence)
        decoded_tags = self._viterbi_decode_manual(lstm_features, mask)
        return decoded_tags

def word2features(sent, i):
    word = sent['tokens'][i]
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
        features['BOS'] = True

    if i < len(sent['tokens'])-1:
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
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent['tokens']))]

def sent2labels(sent):
    return [tag for tag in sent['ner_tags']]

# --- Функции из ner_bert_kaggle.py ---
MAX_SEQ_LENGTH = 128

def tokenize_and_align_labels_bert(sentences: List[dict], tokenizer: BertTokenizerFast, max_length: int, tag_to_id: dict, id_to_tag: dict):
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

def add_tag_ids_to_sentences(sentences: List[dict], tag_to_id_map: dict) -> List[dict]:
    for sentence in sentences:
        sentence['ner_tags_ids'] = [tag_to_id_map.get(tag, tag_to_id_map['O']) for tag in sentence['ner_tags']]
    return sentences

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

    # 2. Определение словарей (уже определены выше)

    # 3. Подготовка данных для Bi-LSTM-CRF
    print("Подготовка данных для Bi-LSTM-CRF...")
    token_to_id, id_to_token, vocab = build_vocab(train_data_raw)
    test_token_ids = tokens_to_ids(test_data_raw, token_to_id)
    test_tag_ids = tags_to_ids(test_data_raw, TAG_TO_ID)

    test_dataset_lstm = NERDataset(test_token_ids, test_tag_ids)
    BATCH_SIZE_LSTM = 32 # Используем тот же размер батча, что и при обучении
    test_dataloader_lstm = DataLoader(test_dataset_lstm, batch_size=BATCH_SIZE_LSTM, shuffle=False, collate_fn=collate_fn)
    print("Данные для Bi-LSTM-CRF подготовлены.")

    # 4. Подготовка данных для BERT
    print("Подготовка данных для BERT...")
    tokenizer_bert = BertTokenizerFast.from_pretrained('bert-base-cased')
    test_data_processed_bert = add_tag_ids_to_sentences(test_data_raw, TAG_TO_ID)
    test_encoded_bert = tokenize_and_align_labels_bert(test_data_processed_bert, tokenizer_bert, MAX_SEQ_LENGTH, TAG_TO_ID, ID_TO_TAG)
    
    BATCH_SIZE_BERT = 16 # Используем тот же размер батча, что и при обучении
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
    
    lstm_crf_model = BiLSTM_CRF(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_TAGS)
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
                sentence_true_labels = [tags[i, j].item() for j in range(tags.size(1)) if mask[i, j].item() == 1]
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
        true_tags = test_data_raw[i]['ner_tags'] # Истинные теги из исходных данных

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