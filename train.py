# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:05:39 2023

@author: YUNDAM
"""

import pandas as pd
import numpy as np
import os

os.chdir(r'C:\Users\YUNDAM\Desktop\075Competition\001Dacon\open')
from sktime.forecasting.model_selection import temporal_train_test_split
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import random
# %% re-load 

train_df = pd.read_csv('./train_processed_230817_2.csv', encoding = 'cp949')

# %% test data pre-processing

test = pd.read_csv('./test_processed_230817_2.csv', encoding = 'cp949')

# %% re-load test

# valid = pd.read_csv('./test_processed_230814.csv', encoding = 'cp949')
train = train_df
valid = test
# train = train_df.loc[train_df['timestamp'] < '2022-08-18']
# valid = train_df.loc[train_df['timestamp'] >= '2022-08-18']
# %% params and functions

xgb_params = {
    "boosting": "gbdt",
    "num_leaves": 1000,
    "eta": 0.05,
    "max_depth": 50,
    "n_estimators": 30000,
    "subsmaple": 0.8
}


params = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'metric': 'rmse',
    'subsample': 1, # 1
    'subsample_freq': 1, # 1
    'num_leaves': 1280, # 6
    'min_data_in_leaf': 6, # 6
    'feature_fraction': 0.8, # 0.5
    'max_bin': 100, # 100,
    'n_estimators': 30000,
}

cat_mae_params = {
    'objective': 'MAE',
    'n_estimators': 30000,
    'early_stopping_rounds': 50, 
    "learning_rate": 0.05,
    "depth": 10
    
}

def weighted_mse(alpha = 1):
    def weighted_mse_fixed(label, pred):
        residual = (label - pred).astype("float")
        grad = np.where(residual>0, -2*alpha*residual, -2*residual)
        hess = np.where(residual>0, 2*alpha, 2.0)
        return grad, hess
    return weighted_mse_fixed

def smape(A, F):
    return 100 / len(A) * np.sum(np.abs(F - A) / (np.abs(A) + np.abs(F)))

nround = 10000

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(42)

# %% model training and prediction



modelList =[2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 
         2, 2, 2, 2, 2, 1, 2, 2, 2, 0,
         2, 2, 2, 2, 2, 2, 1, 1, 2, 0,
         1, 0, 2, 0, 0, 2, 2, 0, 2, 2,
         1, 1, 2, 1, 0, 2, 1, 1, 0, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 1,
         1, 0, 2, 2, 1, 1, 1, 1, 1, 2,
         2, 2, 1, 2, 2, 2, 2, 2, 2, 2,
         1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
         2, 0, 0, 0, 2, 2, 1, 0, 1, 2] # model selection




modelList =[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
         2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
         2, 0, 2, 0, 0, 2, 2, 0, 2, 2,
         2, 2, 2, 2, 0, 2, 2, 2, 0, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 0, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 0, 0, 0, 2, 2, 2, 0, 2, 2] # model selection



predList = []
smapeList = []

categorical_features = ["weekend", "weekday"]
paramsList = []
# for b in range(100):
for b in range(67, 100):
# for b in [53, 13, 39, 41, 64, 89, 90, 94, 97]:   
    
    print('building Number : {}'.format(b+1))   
        
    trainBldg = train[train['건물번호'] == b+1]
    validBldg = valid[valid['건물번호'] == b+1]
    
    # trainBldg = trainBldg.reindex(range(len(trainBldg)))
    # validBldg = validBldg.reindex(range(len(validBldg)))    
    # trainFeatures = trainBldg.drop(['전력소비량(kWh)','timestamp', '일조(hr)', '일사(MJ/m2)', '건물번호'], axis = 1)
    trainFeatures = trainBldg.drop(['전력소비량(kWh)','timestamp', '일조(hr)', '일사(MJ/m2)', '건물번호', 'is_holiday','wci', 'cos_time','풍속(m/s)','습도(%)'], axis = 1)    
    trainTarget = np.log1p(trainBldg['전력소비량(kWh)'])
    
    trainFeatures = np.array(trainFeatures)
    trainTarget = np.array(trainTarget)
    # validFeatures = validBldg.drop(['전력소비량(kWh)','timestamp', '일조(hr)', '일사(MJ/m2)', '건물번호'], axis = 1)
    # validFeatures = validBldg.drop(['전력소비량(kWh)','timestamp', '일조(hr)', '일사(MJ/m2)', '건물번호','is_holiday','wci', 'cos_time','풍속(m/s)','습도(%)'], axis = 1)
    validFeatures = validBldg.drop(['timestamp', '건물번호','is_holiday','wci', 'cos_time','풍속(m/s)','습도(%)'], axis=1)
    # validTarget = np.log1p(validBldg['전력소비량(kWh)'])
    
    validFeatures = np.array(validFeatures)
    # validTarget = np.array(validTarget)
    
    y_train, y_valid, x_train, x_valid = temporal_train_test_split(y = trainTarget, X = trainFeatures, test_size = 168)
    
    kf = KFold(n_splits=3)
    models = []
    
    first = 0
    for train_index, test_index in kf.split(trainFeatures):
        
        train_features = trainFeatures[train_index]
        train_target = trainTarget[train_index]
    
        test_features = trainFeatures[test_index]
        test_target = trainTarget[test_index]

        if modelList[b] == 0:
            print('building Number : {}'.format(b+1))
            if first == 0:                 
                sampler = TPESampler(seed=0)              
                print('building Number : {}'.format(b+1))    
                def XGBobjective(trial):

                    param = {
                        'objective': 'MAE',
                        'metric': 'mape', 
                        'max_depth': trial.suggest_int('max_depth',5, 100),
                        'eta': trial.suggest_categorical("eta", [0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07]),
                        'n_estimators': trial.suggest_categorical('n_estimators', [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]),
                        'subsample': trial.suggest_loguniform('subsample', 0.6, 1),
                    }
                    
                    model = XGBRegressor(**param)
                    model.set_params(**{'objective':weighted_mse(100), 'early_stopping_rounds': 25})
                    lgb_model = model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
                    perf = smape(y_valid, lgb_model.predict(x_valid))
                    return perf
                        
                study_lgb = optuna.create_study(direction='minimize', sampler=sampler)
                study_lgb.optimize(XGBobjective, n_trials=25)
                trial = study_lgb.best_trial
                Xgbmparams = trial.params
                paramsList.append(Xgbmparams)
                first += 1       
            
            print('building Number : {}'.format(b+1))    
            #xbg 학습 
            model=XGBRegressor(**Xgbmparams)
            model.set_params(**{'objective':weighted_mse(100), 'early_stopping_rounds': 25})            
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)   
            models.append(model)


        elif modelList[b] == 1:
            print('building Number : {}'.format(b+1))    
            if first == 0:                 
                sampler = TPESampler(seed=0)              
                print('building Number : {}'.format(b+1))
                def LGBMobjective(trial):

                    param = {
                        'objective': 'MAE',
                        'metric': 'mape', 
                        'max_depth': trial.suggest_int('max_depth',5, 100),
                        'learning_rate': trial.suggest_categorical("learning_rate", [0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07]),
                        'n_estimators': trial.suggest_categorical('n_estimators', [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'subsample': trial.suggest_loguniform('subsample', 0.6, 1),
                        'num_leaves': trial.suggest_int('num_leaves', 100, 2000)
                    }
                    
                    model = lgb.LGBMRegressor(**param)
                    lgb_model = model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=25)
                    perf = smape(y_valid, lgb_model.predict(x_valid))
                    return perf
                        
                study_lgb = optuna.create_study(direction='minimize', sampler=sampler)
                study_lgb.optimize(LGBMobjective, n_trials=25)
                trial = study_lgb.best_trial
                Lgbmparams = trial.params
                paramsList.append(Lgbmparams)
                first += 1       
                
            print('building Number : {}'.format(b+1))    
            #lgbm 학습 
            model=lgb.LGBMRegressor(**Lgbmparams)
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])   
            models.append(model)
            

        elif modelList[b] == 2:
            print('building Number : {}'.format(b+1))
            if first == 0:                 
                sampler = TPESampler(seed=0)
                print('building Number : {}'.format(b+1))
                def CATobjective(trial):
        
                    param = {
                        'objective': 'MAE',
                        'depth': trial.suggest_int('depth',2, 16),
                        'learning_rate': trial.suggest_categorical("learning_rate", [0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07]),
                        'n_estimators': trial.suggest_categorical('n_estimators', [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]),
                    }
                    
        
                    
                    model = CatBoostRegressor(**param)
                    cat_model = model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=25)
                    perf = smape(y_valid, cat_model.predict(x_valid))
                    return perf
                        
                study_lgb = optuna.create_study(direction='minimize', sampler=sampler)
                study_lgb.optimize(CATobjective, n_trials=10)
                trial = study_lgb.best_trial
                Catparams = trial.params
                paramsList.append(Catparams)
                first += 1       
            
            print('building Number : {}'.format(b+1))
            model = CatBoostRegressor(**Catparams)
            model.fit(train_features, train_target, eval_set=[(test_features, test_target)], early_stopping_rounds=25)
            models.append(model)

    results = []

    for model in models:

        if  len(results) == 0:
            results = np.expm1(model.predict(validFeatures)) / len(models)
            
        else:
            results += np.expm1(model.predict(validFeatures)) / len(models)
    print('building Number : {}'.format(b+1))        
    predList.extend([x for x in results])
    # smapeList.append(smape(np.expm1(validTarget), results))

# %%

predFinal = np.expm1(predList)
# print('Avg. SMAPE : {}'.format(np.mean(smapeList)))

# # %% post-processing

# from copy import deepcopy
# predSave = deepcopy(predFinal)


# post = []
# for k in range(100):
    
#     base = np.min(train.loc[train['건물번호'] == k+1 , '전력소비량(kWh)'])
#     predBase = np.min(predFinal[168*k:168*(k+1)])
    
#     if base <= predBase:
        
#         post.append(1)
        
#     else:
#         post.append(0)
#         minFinder = predFinal[168*k:168*(k+1)]
#         minIndex = minFinder.index(np.min(minFinder))
#         predFinal[168*k + minIndex] = base
        

a = pd.DataFrame(data=None)

# %% submission

sample_submission = pd.read_csv(r'./sample_submission.csv')
sample_submission['answer'] = predFinal
sample_submission.to_csv('lgb_submission_230819_1.csv', index=False)
