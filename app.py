# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import itertools
    import warnings
    import matplotlib.pyplot as plt 
    from statsmodels.tsa.arima_model import ARIMA
    from pmdarima.arima import ndiffs
    from pmdarima.arima import ADFTest
    from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
    from pmdarima.arima import auto_arima
    
    # 讀檔
    df_training = pd.read_csv(args.training)

    # 先將索引轉成時間
    df_training["time"] = pd.to_datetime(df_training["time"])
    df_training = df_training.set_index("time")
   
    
    '''
    df_training.plot(figsize=(15, 8))
    plt.xlabel('Time', fontsize = 20)
    plt.ylabel('operating_reserve', fontsize = 20)
    plt.show()
    '''
    
    arima_result = auto_arima(df_training["operating_reserve"]).predict(n_periods=15)
    arima_result = arima_result*10
    arima_result = {"date": [20220330, 20220331, 20220401, 20220402, 20220403, 20220404, 20220405, 
             20220406, 20220407, 20220408, 20220409, 20220410, 20220411, 20220412, 20220413],
             "operating_reserve(MW)": arima_result[-15:]
             }
    df_result = pd.DataFrame(arima_result)
    df_result.to_csv(args.output, index=0)
    