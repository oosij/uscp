# import list

from transformers import ElectraTokenizer

## 오늘 날짜 확인
from datetime import datetime, timedelta

from us_news_sc_model import model_select
from us_news_sc_ssh import ssh_access_day_files
from us_news_sc_predict import sentiments_score_extract
from us_news_sc_predict import stock_dic_make, ticker_output, day_output


model_path = 'google/electra-base-discriminator'  # "monologg/koelectra-base-v3-discriminator"
tokenizer =  ElectraTokenizer.from_pretrained(model_path )

select = 'all' # 'half' is 50agree , 'all' is allagree -> 100
model = model_select(select)
#print(model)

MAX_LEN, batch_size = 64, 16


# ssh info
host  = '121.254.150.83'
port_num  = 6502
user_id = 'bascrap'
user_password = 'qwe123!@#'

# target date
yesterday = datetime.today() - timedelta(1)
yesterday_date = yesterday.strftime("%Y%m%d")
## 오전 10 ~ 2시 사이에 업데이트?
yesterday_date = '20230325'  # 테스트용  -> 날짜 지정 YYYYMMDD

# ssh 타겟 경로
dir_path = '/home/bascrap/data/stock/investing_news/'
folder_path = yesterday_date + '/'
remote_dir_path = dir_path + folder_path

## 파일 불러오기
target_list, ssh = ssh_access_day_files(host, port_num, user_id, user_password, remote_dir_path)
result_list = sentiments_score_extract(target_list, remote_dir_path, ssh, model, tokenizer, MAX_LEN, batch_size, yesterday_date)


## 종목, 일짜 합산 평균
stock_dic = stock_dic_make(result_list)
ticker_df = ticker_output(stock_dic)  # 종목 출력
day_avg_preds = day_output(stock_dic)  #하루 출력
day_df = yesterday_date, day_avg_preds

print(len(result_list))
print(ticker_df)
print(day_df)
