
# coding: utf-8

# # Improvement: separate classifiers for lobby 1 and  lobbies 0, 7

# In[1]:

import pandas as pd


# In[2]:

import numpy as np


# In[3]:

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import scale

from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

from sklearn.metrics import roc_auc_score


# # The function that cleans data

# In[4]:

# Input: pandas dataframe, Output: numpy matrix
def clean_data(features) :
    
    features.drop('lobby_type', axis = 1, inplace = True)
    
    # Time
    min_time = min(features['start_time'])
    features['start_time'] = features['start_time'] - min(features['start_time'])
    
    # NaN's: categorical
    categorical_features = ['first_blood_player1', 'first_blood_player2',
                            'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                            'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
    for categorical_feature in categorical_features :
        most_popular = features[categorical_feature].dropna().value_counts().idxmax()
        features[categorical_feature].fillna(most_popular, inplace = True)
        
    # NaN's: numerical
    time_features = ['first_blood_time',
                     'radiant_bottle_time', 'radiant_courier_time', 'radiant_flying_courier_time', 'radiant_first_ward_time',
                     'dire_bottle_time', 'dire_courier_time', 'dire_flying_courier_time', 'dire_first_ward_time']
    
    numerical_features = []
    for feature in features :
        if (feature not in categorical_features) :
            numerical_features.append(feature)
            
    for numerical_feature in numerical_features :
        if (numerical_feature in time_features) :
            features[numerical_feature].fillna(300, inplace = True)
        else :
            median = features[numerical_feature].dropna().median()
            features[numerical_feature].fillna(median, inplace = True)
    
    # Scaling numerical features
    X_numerical_scaled = scale(features[numerical_features].as_matrix())
    
    # How many heros are in the game?
    hero_features = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                     'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
    n_heroes = [max(features_train[x].value_counts().index) for x in hero_features]
    N = max(n_heroes)
      
    # Using bag of words for categorical features
    X_pick = np.zeros((features.shape[0], N))
    for i, match_id in enumerate(features.index):
        for p in xrange(5):
            X_pick[i, features.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            X_pick[i, features.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    
    return np.hstack((X_numerical_scaled, X_pick))


# # Getting and preparing data

# In[24]:

features_train = pd.read_csv('./data/features.csv', index_col='match_id')
features_test  = pd.read_csv('./data/features_test.csv', index_col='match_id')


# In[25]:

target_train = features_train['radiant_win']


# In[26]:

features_train.drop(['duration', 'radiant_win', 'tower_status_radiant','tower_status_dire',
                     'barracks_status_radiant', 'barracks_status_dire'], axis = 1, inplace = True)


# In[27]:

features_train['lobby_type'].value_counts()


# In[28]:

features_train_lobby1  = features_train[features_train['lobby_type'] == 1].copy()
features_train_lobby07 = features_train[features_train['lobby_type'] != 1].copy()

features_test_lobby1  = features_test[features_test['lobby_type'] == 1].copy()
features_test_lobby07 = features_test[features_test['lobby_type'] != 1].copy()


# In[29]:

X_train_lobby1  = clean_data(features_train_lobby1)
X_train_lobby07 = clean_data(features_train_lobby07)

X_test_lobby1  = clean_data(features_test_lobby1)
X_test_lobby07 = clean_data(features_test_lobby07)


# In[30]:

y_train_lobby1  = target_train[features_train['lobby_type'] == 1].as_matrix()
y_train_lobby07 = target_train[features_train['lobby_type'] != 1].as_matrix()


# # Creating and testing logistic regression model

# In[31]:

clf_lobby1 = LogisticRegression(penalty = 'l2', C = 0.1, random_state = 42, n_jobs = -1)
clf_lobby1.fit(X_train_lobby1, y_train_lobby1)


# In[ ]:

clf_lobby07 = LogisticRegression(penalty = 'l2', C = 0.1, random_state = 42, n_jobs = -1)
clf_lobby07.fit(X_train_lobby07, y_train_lobby07)


# In[ ]:

kfold = KFold(len(y_train_lobby1), n_folds = 5, shuffle = True, random_state = 42) 
score = cross_val_score(estimator = clf_lobby1,
                        X = X_train_lobby1,
                        y = y_train_lobby1,
                        cv = kfold,
                        scoring = 'roc_auc',
                        n_jobs = -1,
                        verbose = True)


# In[ ]:

print score.mean()


# In[ ]:

kfold = KFold(len(y_train_lobby07), n_folds = 5, shuffle = True, random_state = 42) 
score = cross_val_score(estimator = clf_lobby07,
                        X = X_train_lobby07,
                        y = y_train_lobby07,
                        cv = kfold,
                        scoring = 'roc_auc',
                        n_jobs = -1,
                        verbose = True)


# In[ ]:

print score.mean()


# In[ ]:

predictions_lobby1 = clf_lobby1.predict_proba(X_test_lobby1)[:, 1]
predictions_lobby1_df = pd.DataFrame({'match_id'    : features_test_lobby1.index,
                                      'radiant_win' : predictions_lobby1
                                     })


# In[ ]:

predictions_lobby07 = clf_lobby07.predict_proba(X_test_lobby07)[:, 1]
predictions_lobby07_df = pd.DataFrame({'match_id'   : features_test_lobby07.index,
                                      'radiant_win' : predictions_lobby07
                                      })


# In[ ]:

kaggle_answer = pd.concat([predictions_lobby1_df, predictions_lobby07_df])


# In[ ]:

kaggle_answer.sort_values(by = ['match_id'], inplace = True)


# In[ ]:

kaggle_answer.to_csv('kaggle_answers/3.csv', index = False)

