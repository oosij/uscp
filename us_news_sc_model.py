# import list
import os
import re
from tqdm import tqdm
import numpy as np
import time
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import ElectraModel, ElectraTokenizer, AutoTokenizer

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# based model

model_path = 'google/electra-base-discriminator'  # "monologg/koelectra-base-v3-discriminator"
model =  ElectraModel.from_pretrained(model_path)  # KoELECTRA-Small-v3
tokenizer =  ElectraTokenizer.from_pretrained(model_path )



## weight model select
def model_select(select):
    # 저장한 모델 가중치 불러오기
    out_dir = './model'
    set_seed(42)  # Set seed for reproducibility
    if select == 'all':
        model = AllAgreeClassifier()
        path = out_dir + '/' + 'saved_weights_2303_acc96(allagree).pt'
    elif select == 'half':
        model = HalfAgreeClassifier()
        path = out_dir + '/' + 'saved_weights_2303_acc86(50agree).pt'
    else: # add or,  default 'all'
        model = AllAgreeClassifier()
        path = out_dir + '/' + 'saved_weights_2303_acc96(allagree).pt'

    model.load_state_dict(torch.load(path))  # , map_location=device   / strict = False
    return model


### 함수들
def text_word_one_limit(t_split):
    con_text = ''
    for tn in range(len(t_split)):
        tns = t_split[tn]
        if len(tns) <= 1:
            continue

        con_text = con_text + ' ' + tns
    return con_text.strip()


# 전처리 함수, 추후 증권 뉴스에 맞게 일부 추가 및 수정 필요
def text_preprocessing(text):
    cleaned_text = re.sub('[^a-zA-Z]', ' ', text)
    cleaned_text = re.sub('\n', '', cleaned_text)
    cleaned_text = re.sub('\xa0', '', cleaned_text)
    cleaned_text = re.sub(
        '[\{\}\[\]\/?;:|\)…－〕.〔ⓘㅇ÷♠♣＜＞©◀Ⅱ·―Ⅱ＆,？☏☎™×『』《》／┌─┬┐│├ ┼┤└┴┘★〈●○[］〉±▨→↑↓∼％「」※ㆍ♥①②③④⑤⑥⑦⑧⑨△◇ ㈜ⓝ◈；：“”‘’ *~【】♡♥▽▷ⓒ▣◇□㈜◆☞■▶▲▼`!^\-_+<>@\#$%&\\\=\(\'\"]',
        ' ', cleaned_text)

    return cleaned_text

# BERT 입력전 전처리
def preprocessing_for_bert(data, tokenizer):
    MAX_LEN = 64
    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            #pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


# ======================================================================

# /model/saved_weights_2303_acc86(50agree).pt
class HalfAgreeClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(HalfAgreeClassifier, self).__init__()
        D_in, H, D_out = 768, 256, 3
        model_path = 'google/electra-base-discriminator'
        self.bert = ElectraModel.from_pretrained(model_path)

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        return logits


# ./model/saved_weights_2303_acc96(allagree).pt
class AllAgreeClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(AllAgreeClassifier, self).__init__()
        D_in, H, D_out = 768, 768, 3
        model_path = 'google/electra-base-discriminator'
        self.bert = ElectraModel.from_pretrained(model_path)

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        return logits


# ======================================================================


def bert_predict(model, test_dataloader):
    device = torch.device('cpu')   # GPU 없을시, CPU로 해야함 단점은 속도가 느림!
    model.eval()
    all_logits = []

    for batch in test_dataloader:
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

