Define : 

해외(미국,영어) 금융 뉴스 학습 데이터셋을 활용한 감정 분류 모델입니다. (부정, 중립, 긍정)

Dataset :

사용한 학습 데이터셋은, financial_phrasebank로, 16명이 주석으로 검수한 데이터셋이며, 모델은 환경에 따라 Electra를 사용했습니다.

Model :

금융 도메인 사전 학습 모델로, FinBERT(로이터 금융 뉴스 기반), FinBERT(재무 분석, 연차보고서 기반) 
범용 사전 학습 모델로, Electra, DeBERTa, XLNet, BigBird 등을 사용하여 비교 분석하였습니다.

Model performence :
![image](https://user-images.githubusercontent.com/94098546/228991999-aa583afa-ba71-480a-a23b-8c1c230e3b81.png)


            50agree (total : 4846)	                            100agree (total : 2264)
            Accuracy	Macro f1	wighted f1	                    Accuracy	Macro f1	wighted f1
FinBERT(1)	0.86	   0.85	      0.87	                        0.98	    0.97	    0.98
FinBERT(2)	0.84	   0.82	      0.84	                        0.97	    0.96	    0.97
Electra	0.86	   0.85	      0.86	                        0.96	    0.96	    0.97
DeBEERTa	0.85	   0.83	      0.85	                        0.98	    0.97	    0.98
XLNet	0.84	   0.83	      0.84	                        0.97	    0.96	    0.97
BigBird	0.86	   0.85	      0.86	                        0.98	    0.97	    0.98

이 중, Electra 모델이 성능이 높게 나온 3 모델 중 비용이 가장 적기에 해당 모델 위주로 다시 최적화 작업을 진행하였습니다.


Pipeline : 

50agree wight 모델과 allagree wight 모델 2가지를 기반으로 실제 raw data가 입력되고 예측되는 것을 구성하였습니다.
실제 raw data는 뉴스 기사, 문서이기에 각 문장을 nltk 라이브러리를 활용해서 전처리 및 문장 분리를 하였고, 각각의 문장을 입력으로 예측 label을 출력하였습니다.
이렇게 나온 예측 label의 값을 합산, 평균으로 simple하게 계산하였습니다. 
예) A 뉴스 기사의 문장수 10 개 -> [문장1, 문장2, ... 문장10]  ->  sum[문장1의 예측 label, 문장2의 예측 label, ... 문장10의 예측 label] / 10


성능적으로 개선사항은 각 문장에 대한 가중치 부여, 또는 문장 구조상 앞문장과 뒷문장의 관계에 대한 부분도 고려하면 좋습니다.
