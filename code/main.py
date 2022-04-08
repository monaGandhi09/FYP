import pandas as pd
import time
import os
import inspect
import multiprocessing as mp
from sklearn.model_selection import train_test_split

from models import RandomForest, SVM, XGBoost
from methods import DSSTE, smote, randomOverSampler, borderlineSmote, adasyn, ensemble_os_all, ensemble_os_selected
from methods import randomUnderSampler, tomekLinks, clusterCentroids, ENN, IHT, ensemble_us_all 

def for_model(X_train, y_train, X_test, y_test, folder, models=[RandomForest]):
    
    df_list = []
    for i, _ in enumerate(models):    
        df = pd.DataFrame(columns=['Technique', 'Accuracy', 'F1 score', 'Precision', 'Recall'])
        df_list.append(df)

    start = time.time()
    # print(f"Getting {model} scores...")

    print("Without Imbalance reduction...")
    for i, model in enumerate(models):
        score = model(X_train, y_train, X_test, y_test)
        score['Technique'] = 'None'
        df_list[i] = df_list[i].append(score, ignore_index=True)
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)
    
    print("Using Smote for imbalance reduction...")
    X_train_smote, y_train_smote = smote(X_train, y_train)
    for i, model in enumerate(models):
        score = model(X_train_smote, y_train_smote, X_test, y_test)
        score['Technique'] = 'Smote'
        df_list[i] = df_list[i].append(score, ignore_index=True)
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)

    print("Using Random Over Sampling for imbalance reduction...")
    X_train_ros, y_train_ros = randomOverSampler(X_train, y_train)
    for i, model in enumerate(models):
        score = model(X_train_ros, y_train_ros, X_test, y_test)
        score['Technique'] = 'Random Over Sampling'
        df_list[i] = df_list[i].append(score, ignore_index=True)
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)

    print("Using Borderline Smote for imbalance reduction...")
    X_train_borderlinesmote, y_train_borderlinesmote = borderlineSmote(X_train, y_train)
    for i, model in enumerate(models):
        score = model(X_train_borderlinesmote, y_train_borderlinesmote, X_test, y_test)
        score['Technique'] = 'Borderline Smote'
        df_list[i] = df_list[i].append(score, ignore_index=True)
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)

    print("Using Adasyn for imbalance reduction...")
    X_train_adasyn, y_train_adasyn = adasyn(X_train, y_train)
    for i, model in enumerate(models):
        score = model(X_train_adasyn, y_train_adasyn, X_test, y_test)
        score['Technique'] = 'Adasyn'
        df_list[i] = df_list[i].append(score, ignore_index=True)
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)

    print("Using Ensemble of all over sampling methods for imbalance reduction...")
    X_train_smote['attack_type'] = y_train_smote
    X_train_ros['attack_type'] = y_train_ros
    X_train_adasyn['attack_type'] = y_train_adasyn
    X_train_borderlinesmote['attack_type'] = y_train_borderlinesmote

    X_train_ensemble = pd.concat([X_train_smote, X_train_ros, X_train_adasyn, X_train_borderlinesmote])
    X_train_ensemble = X_train_ensemble.drop_duplicates()

    y_train_ensembleosall = X_train_ensemble['attack_type']
    X_train_ensembleosall = X_train_ensemble.iloc[:,:-1]

    for i, model in enumerate(models):
        score = models(X_train_ensembleosall, y_train_ensembleosall, X_test, y_test)
        score['Technique'] = 'Ensemble OS (all)'
        df_list[i] = df_list[i].append(score, ignore_index=True)
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)

    del X_train_adasyn, X_train_borderlinesmote, y_train_smote, y_train_ros, y_train_adasyn, y_train_borderlinesmote, X_train_ensemble, X_train_ensembleosall, y_train_ensembleosall

    print("Using Ensemble of selected over sampling methods for imbalance reduction...")
    X_train_ensemble = pd.concat([X_train_smote, X_train_ros])
    X_train_ensemble = X_train_ensemble.drop_duplicates()

    y_train_ensembleosselected = X_train_ensemble['attack_type']
    X_train_ensembleosselected = X_train_ensemble.iloc[:,:-1]

    for i, model in enumerate(models):
        score = model(X_train_ensembleosselected, y_train_ensembleosselected, X_test, y_test)
        score['Technique'] = 'Ensemble OS (selected)'
        df_list[i] = df_list[i].append(score, ignore_index=True)
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)

    print("Using Random Under Sampling for imbalance reduction...")
    X_train_rus, y_train_rus = randomUnderSampler(X_train, y_train)
    for i, model in enumerate(models):
        score = model(X_train_rus, y_train_rus, X_test, y_test)
        score['Technique'] = 'Random Under Sampling'
        df_list[i] = df_list[i].append(score, ignore_index=True)
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)

    print("Using Tomek Links for imbalance reduction...")
    X_train_tomeklinks, y_train_tomeklinks = tomekLinks(X_train, y_train)
    for i, model in enumerate(models):
        score = model(X_train_tomeklinks, y_train_tomeklinks, X_test, y_test)
        score['Technique'] = 'Tomek Links'
        df_list[i] = df_list[i].append(score, ignore_index=True)
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)

    print("Using Cluster Centroids for imbalance reduction...")
    X_train_cc, y_train_cc = clusterCentroids(X_train, y_train)
    for i, model in enumerate(models):
        score = model(X_train_cc, y_train_cc, X_test, y_test)
        score['Technique'] = 'Cluster Centroids'
        df_list[i] = df_list[i].append(score, ignore_index=True)
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)

    print("Using Edited Nearest Neighbor for imbalance reduction...")
    X_train_enn, y_train_enn = ENN(X_train, y_train)
    for i, model in enumerate(models):
        score = function(X_train_enn, y_train_enn, X_test, y_test)
        score['Technique'] = 'Edited Nearest Neighbor'
        df_list[i] = df_list[i].append(score, ignore_index=True)
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)

    print("Using Instance Hardness Threshold for imbalance reduction...")
    X_train_iht, y_train_iht = IHT(X_train, y_train)
    for i, model in enumerate(models):
        score = model(X_train_iht, y_train_iht, X_test, y_test)
        score['Technique'] = 'Instance Hardness Threshold'
        df_list[i] = df_list[i].append(score, ignore_index=True)
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)

    print("Using Ensemble of all under sampling methods for imbalance reduction...")
    X_train_rus['attack_type'] = y_train_rus
    X_train_tomeklinks['attack_type'] = y_train_tomeklinks
    X_train_cc['attack_type'] = y_train_cc
    X_train_enn['attack_type'] = y_train_enn
    X_train_iht['attack_type'] = y_train_iht

    X_train_ensemble = pd.concat([X_train_rus, X_train_tomeklinks, X_train_cc, X_train_enn, X_train_iht])
    X_train_ensemble = X_train_ensemble.drop_duplicates()

    y_train_ensembleusall = X_train_ensemble['attack_type']
    X_train_ensembleusall = X_train_ensemble.iloc[:,:-1]
    for i, model in enumerate(models):
        score = model(X_train_ensembleusall, y_train_ensembleusall, X_test, y_test)
        score['Technique'] = 'Ensemble US (all)'
        df_list[i] = df_list[i].append(score, ignore_index=True)
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)

    print("Using DSSTE algorithm for imbalance reduction...")
    X_train_dsste, y_train_dsste = DSSTE(X_train, y_train)
    for i, model in enumerate(models):
        score = model(X_train_dsste, y_train_dsste, X_test, y_test)
        score['Technique'] = 'DSSTE'
        df_list[i] = df_list[i].append(score, ignore_index=True)
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)

    end = time.time()
    print(f"Total time taken: {end-start}")

    for i, model in enumerate(models):
        df_list[i].to_csv(f'{folder}/{model.__name__}_scores.csv', index=False)

def nslkdd():

    # TODO: Change the path of the file as required
    dataset = pd.read_csv("../data/nsl-kdd-preprocessed.csv")
    dataset = dataset.drop(columns=['label'])
    
    # split into train and test
    train = dataset.loc[:125972,:]  # 125972 -- size of the training data after preprocessing
    test = dataset.loc[125973:,:]

    X_train = train.drop(columns=['attack_type'])
    y_train = train['attack_type']
    X_test = test.drop(columns=['attack_type'])
    y_test = test['attack_type']

    folder = '../results/nslkdd'
    if 'nslkdd' not in os.listdir('../results'):
        os.makedirs(folder)
    
    models = [RandomForest, SVM, XGBoost]
    for_model(X_train, y_train, X_test, y_test, folder, models)
    
def CSE2018():

    # TODO: Change the path of the file for 2018 dataset
    dataset = pd.read_csv("../data/cic-ids-2018-preprocessed.csv")
    dataset = dataset.drop(columns=['label'])
    
    X = dataset.drop(columns=['category'])
    y = dataset['category']
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    folder = '../results/2018'
    if '2018' not in os.listdir('../results'):
        os.makedirs(folder)

    models = [RandomForest, SVM, XGBoost]
    for_model(X_train, y_train, X_test, y_test, models)
    
if __name__ == '__main__':
    # To get results for nslkdd dataset
    nslkdd()

    # To get results for 2018 dataset
    # CSE2018()