import pandas as pd  # 데이터를 로드하기 위한 라이브러리
import numpy as np
from tensorflow.python.keras.models import Sequential, InputLayer
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from matplotlib import pyplot as plt  # plotting 하기 위한 라이브러리
from sklearn import preprocessing  # 데이터를 전처리하기 위한 라이브러리
from sklearn.model_selection import train_test_split
import sys, os

sys.path.append(os.pardir)
from sklearn.preprocessing import *


''' csv 파일을 load해서 반환하는 함수 '''
def load_CSV(file):
    data = []
    data = pd.read_csv(file)
    return data

def plot_data(x,y,str1,str2):
    plt.scatter(x, y)
    plt.title("Scatter Plot of the data")
    plt.xlabel(str1)
    plt.ylabel(str2)
    plt.show()

def plot_hist(x,label):
    plt.hist(x)
    plt.xlabel(label)
    plt.show()


''' 각 파일의 이름 변수에 data 로드 '''
cus_info = load_CSV('cus_info.csv')  # 고객 및 주거래 계좌 정보
iem_info = load_CSV('iem_info_20210902.csv')  # 종목 정보 _주식 종목에 대한 코드 정보
stk_bnc_hist = load_CSV('stk_bnc_hist.csv')  # 국내 주식 잔고 이력 _일별 종목 잔고수량 및 금액, 액면가 정보
stk_hld_test = load_CSV('stk_hld_test.csv')  # 국내 주식 보유 기간(train) _고객에게 제공되는 과거 국내주식 보유기간 데이터 (681,472건)
stk_hld_train = load_CSV('stk_hld_train.csv')  # 국내 주식 보유 기간(test) _개발한 알고리즘 검증을 위한 문제지 (70,596건)
submission=load_CSV('sample_submission.csv')

''' 필요없는 컬럼 추출 '''
iem_info.drop(['iem_krl_nm'], axis=1, inplace=True) # '종목 한글 명' 삭제


''' 일부 컬럼을 빼서 정규화 시키고 다시 결합해서 원상태의 컬럼으로 결합 '''
# 정규화할 컬럼 추출
bnc_hist_norm = stk_bnc_hist[['bnc_qty', 'tot_aet_amt', 'stk_par_pr']]
stk_bnc_hist.drop(['bnc_qty', 'tot_aet_amt', 'stk_par_pr'], axis=1, inplace=True)

# 데이터 정규화 코드
transformer = MinMaxScaler()
transformer.fit(bnc_hist_norm)
bnc_hist_norm = transformer.transform(bnc_hist_norm)

bnc_hist_norm = pd.DataFrame(bnc_hist_norm)
bnc_hist_norm.columns = (['bnc_qty', 'tot_aet_amt', 'stk_par_pr'])  # 컬럼에 레이블명 지정

# 정규화한 컬럼을 기존 DataFrame과 수평 결합
stk_bnc_hist = pd.concat([stk_bnc_hist, bnc_hist_norm], axis=1)


''' csv파일 결합 : cus_info + stk_bnc_hist + iem_info '''
# act_id를 기준으로 stk_bnc_hist, cus_info 결합
merge_cus_info=pd.merge(cus_info,stk_bnc_hist ,on='act_id')

mcf_df=pd.DataFrame(merge_cus_info)
#mcf_df.to_csv("merge_cus_info.csv",index=False)


# iem_cd를 기준으로 iem_info까지 결합 _ 총 3개 csv파일 결합
merge_data=pd.merge(merge_cus_info,iem_info,on='iem_cd')

merge_df=pd.DataFrame(merge_data)
#merge_df.to_csv("merge_data.csv",index=False)



''' train data와 test data 각각 merge_info와 결합 '''
# 계좌ID, 종목코드를 기준으로 train data를 결합
merge_train=pd.merge(merge_data,stk_hld_train, how="right",left_on=['act_id','iem_cd'],right_on=['act_id','iem_cd'])
merge_train=merge_train[merge_train["byn_dt"]==merge_train["bse_dt"]]
#merge_train=pd.merge(merge_data,stk_hld_train,on=['act_id','iem_cd'])

# 계좌ID, 종목코드를 기준으로 test data를 결합
merge_test = pd.merge(merge_data,stk_hld_test, how="right",left_on=['act_id','iem_cd'],right_on=['act_id','iem_cd'])
merge_test=merge_test[merge_test["byn_dt"]==merge_test["bse_dt"]]
#merge_test=pd.merge(merge_data,stk_hld_test,on=['act_id','iem_cd'])


# 결합한 train data를 csv파일로 저장
train_df=pd.DataFrame(merge_train)
#train_df.to_csv("train_data.csv",index=False)


# 결합한 test data를 csv 파일로 저장
test_df = pd.DataFrame(merge_test)

b_size = 60000 # 대량의 data를 처리하기 위한 mini batch



# hold_d 분리
y_train = train_df[['hold_d']]
x_train = train_df.drop(['hold_d','iem_cd','act_id'],axis=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=True)


ai_model = Sequential([
            InputLayer(input_shape=(17,)),
            Dense(9, activation='relu', name='hidden_layer'),
            Dense(1, activation='sigmoid', name='output_layer')
            ])

ai_model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['accuracy'])
ai_res = ai_model.fit(x_train, y_train, epochs=10, batch_size=b_size, validation_data=(x_val,y_val))


test_df=test_df.drop(['act_id','iem_cd','submit_id','hist_d','hold_d'],axis=1)
test_data=np.array(test_df)

models=[]
models.append(ai_model)

result = []
for i in models:
    result.append(i.predict(test_data))

predict = np.mean(result, axis = 0)
print(predict)
submission["hold_d"] = np.round(predict)
print(submission)
submission.to_csv("dacon_baseline.csv", index = False)
