# DSAI-2022-HW1
## Objective:
利用歷史資料，預測 2022/3/30 ~ 2022/4/13 的備轉容量

## Dataset:
1. 台灣電力公司_過去電力供需資訊2021
2. 台灣電力公司_本年度每日尖峰備轉容量率

## Method:
一開始用Arima做預測  
其自帶差分項，較能處理非穩定的資料，將資料做平穩化  
但其參數(p,d,q)需使用ADF檢驗法做來選取最佳參數  
人工不斷測試十分耗時  
因此後來改採用AutoArima  
可以節省調整參數之時間  
之後陸續有試過Sarima、LSTM等方法  
但效果並未有顯著改善  
因此最後決定採用AutoArima預測的結果  

## Run:
python app.py --training training_data.csv --output submission.csv

