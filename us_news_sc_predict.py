import numpy as np
import pandas as pd

from nltk import sent_tokenize
from us_news_sc_ssh import content_extract
from us_news_sc_model import text_word_one_limit, text_preprocessing, preprocessing_for_bert, bert_predict
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

## SSH 연결
import paramiko


def sentiments_score_extract(target_list, remote_dir_path, ssh, model, tokenizer, MAX_LEN, batch_size, yesterday_date):
    result_list = []

    for number in range(len(target_list)):
        symbol, title =  target_list[number].split('_')[0], target_list[number].split('_')[1]
        remote_file_path = remote_dir_path +  target_list[number]
        body = content_extract(remote_file_path, ssh)
        tokenized_text = sent_tokenize(body)
        text_list = [title] + tokenized_text
        doc_tokens = preprocessing_contents(text_list)
        preds = prediction_labels(model, doc_tokens, batch_size, tokenizer)
        sum_preds = sum(preds)
        avg_preds = sum(preds )/ len(preds)
        # 종목, 제목, 본문, 문장별 예측라벨, 예측 합 , 예측 평균 , 날짜
        data = [symbol, title, body, preds ,sum_preds, avg_preds, yesterday_date]
        result_list.append(data)
    ssh.close()
    return result_list


# 모델 입력전 전처리 함수
def preprocessing_contents(text_list):
    doc_tokens = []

    for d in range(len(text_list)):
        X_pred = text_list[d]
        X_pred = text_preprocessing(X_pred)
        X_pred_split = X_pred.split(' ')
        X_pred = text_word_one_limit(X_pred_split)
        doc_tokens.append(X_pred)

    return doc_tokens


# 모델에 예측 라벨을 계산하는 함수
def prediction_labels(model, doc_tokens, batch_size, tokenizer):
    pred_inputs, pred_masks = preprocessing_for_bert(doc_tokens, tokenizer)
    pred_dataset = TensorDataset(pred_inputs, pred_masks)
    pred_sampler = SequentialSampler(pred_dataset)
    pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=batch_size)

    probs = bert_predict(model, pred_dataloader)
    preds = np.argmax(probs, axis=1)
    preds = list(preds)

    results = []

    for p in range(len(preds)):
        p_plus = preds[p]
        p_real = p_plus - 1
        results.append(p_real)

    return results


# output 을 위한 딕셔너리 구성
def stock_dic_make(result_list):
    stock_dic = {}
    for r in range(len(result_list)):
        in_data = result_list[r]
        symbol, avg_preds = in_data[0], in_data[5]
        try:
            stock_dic[symbol].append(avg_preds)
        except:  # 처음꺼라면
            stock_dic[symbol] = []
            stock_dic[symbol].append(avg_preds)
    return stock_dic


## 종목별 감정 스코어 평균
def ticker_output(stock_dic):
    ## output
    ticker_list = list(stock_dic.keys())
    ticker_data = []

    for tn in range(len(ticker_list)):
        ticker_name = ticker_list[tn]
        stock_avg_preds = sum(stock_dic[ticker_name]) / len(stock_dic[ticker_name])
        in_ticker = [ticker_name, stock_avg_preds]
        ticker_data.append(in_ticker)

    ticker_df = pd.DataFrame(ticker_data, columns=['symbol', 'sentiment score'])
    return ticker_df


## 일짜 전체 출력 : 여기까지 하고 잠깐 스터디!
def day_output(stock_dic):
    ticker_list = list(stock_dic.keys())
    total_day = []

    for t in range(len(ticker_list)):
        ticker = ticker_list[t]
        ticker_avg_preds = sum(stock_dic[ticker]) / len(stock_dic[ticker])
        total_day.append(ticker_avg_preds)
    day_avg_preds = sum(total_day) / len(ticker_list)

    return day_avg_preds