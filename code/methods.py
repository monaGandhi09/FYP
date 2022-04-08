import pandas as pd
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE, KMeansSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids, EditedNearestNeighbours, InstanceHardnessThreshold

def smote(X_train, y_train):
    model = SMOTE()
    X_train, y_train = model.fit_resample(X_train, y_train)
    return X_train, y_train

def randomOverSampler(X_train, y_train):
    model = RandomOverSampler()
    X_train, y_train = model.fit_resample(X_train, y_train)
    return X_train, y_train

def borderlineSmote(X_train, y_train):
    model = BorderlineSMOTE()
    X_train, y_train = model.fit_resample(X_train, y_train)
    return X_train, y_train

def adasyn(X_train, y_train):
    model = ADASYN()
    X_train, y_train = model.fit_resample(X_train, y_train)
    return X_train, y_train

def ensemble_os_all(X_train, y_train):

    X_train_smote, y_train_smote = smote(X_train, y_train)
    X_train_smote['attack_type'] = y_train_smote
    X_train_ros, y_train_ros = randomOverSampler(X_train, y_train)
    X_train_ros['attack_type'] = y_train_ros
    X_train_adasyn, y_train_adasyn = adasyn(X_train, y_train)
    X_train_adasyn['attack_type'] = y_train_adasyn
    X_train_borderlineSmote, y_train_borderlineSmote = borderlineSmote(X_train, y_train)
    X_train_borderlineSmote['attack_type'] = y_train_borderlineSmote

    X_train_ensemble = pd.concat([X_train_smote, X_train_ros, X_train_adasyn, X_train_borderlineSmote])
    X_train_ensemble = X_train_ensemble.drop_duplicates()

    y_train_ensemble = X_train_ensemble['attack_type']
    X_train_ensemble = X_train_ensemble.iloc[:,:-1]

    return X_train_ensemble, y_train_ensemble

def ensemble_os_selected(X_train, y_train):

    X_train_smote, y_train_smote = smote(X_train, y_train)
    X_train_smote['attack_type'] = y_train_smote
    X_train_ros, y_train_ros = randomOverSampler(X_train, y_train)
    X_train_ros['attack_type'] = y_train_ros

    X_train_ensemble = pd.concat([X_train_smote, X_train_ros])
    X_train_ensemble = X_train_ensemble.drop_duplicates()

    y_train_ensemble = X_train_ensemble['attack_type']
    X_train_ensemble = X_train_ensemble.iloc[:,:-1]

    return X_train_ensemble, y_train_ensemble

def randomUnderSampler(X_train, y_train):
    model = RandomUnderSampler(sampling_strategy='majority')
    X_train, y_train = model.fit_resample(X_train, y_train)
    return X_train, y_train

def tomekLinks(X_train, y_train):
    model = TomekLinks()
    X_train, y_train = model.fit_resample(X_train, y_train)
    return X_train, y_train

def clusterCentroids(X_train, y_train):
    model = ClusterCentroids()
    X_train, y_train = model.fit_resample(X_train, y_train)
    return X_train, y_train

def ENN(X_train, y_train):
    model = EditedNearestNeighbours()
    X_train, y_train = model.fit_resample(X_train, y_train)
    return X_train, y_train

def IHT(X_train, y_train):
    model = InstanceHardnessThreshold()
    X_train, y_train = model.fit_resample(X_train, y_train)
    return X_train, y_train

def ensemble_us_all(X_train, y_train):

    X_train_rus, y_train_rus = randomUnderSampler(X_train, y_train)
    X_train_rus['attack_type'] = y_train_rus
    X_train_tomeklinks, y_train_tomeklinks = tomekLinks(X_train, y_train)
    X_train_tomeklinks['attack_type'] = y_train_tomeklinks
    X_train_cc, y_train_cc = clusterCentroids(X_train, y_train)
    X_train_cc['attack_type'] = y_train_cc
    X_train_enn, y_train_enn = ENN(X_train, y_train)
    X_train_enn['attack_type'] = y_train_enn
    X_train_iht, y_train_iht = IHT(X_train, y_train)
    X_train_iht['attack_type'] = y_train_iht

    X_train_ensemble = pd.concat([X_train_rus, X_train_tomeklinks, X_train_cc, X_train_enn, X_train_iht])
    X_train_ensemble = X_train_ensemble.drop_duplicates()

    y_train_ensemble = X_train_ensemble['attack_type']
    X_train_ensemble = X_train_ensemble.iloc[:,:-1]

    return X_train_ensemble, y_train_ensemble

def DSSTE(X_train, y_train):

    enn = EditedNearestNeighbours(n_neighbors=50, sampling_strategy='all')
    X_res, y_res = enn.fit_resample(X_train, y_train)

    easy_set = X_res.copy(deep=True)
    easy_set['attack_type'] = y_res
    train = X_train.copy(deep=True)
    train['attack_type'] = y_train
    difficult_set = pd.concat([train, easy_set]).drop_duplicates(keep=False)

    difficult_set_maj = difficult_set[(difficult_set['attack_type'] == 0) | (difficult_set['attack_type'] == 4) | (difficult_set['attack_type'] == 1) ]
    difficult_set_min = difficult_set[(difficult_set['attack_type'] == 2)| (difficult_set['attack_type'] == 3)]

    X_2 = difficult_set_maj.drop(columns=['attack_type'])
    y_2 = difficult_set_maj['attack_type']

    kmeans = KMeans(n_clusters=50, random_state=42)

    cc = ClusterCentroids(random_state=42, estimator=kmeans, sampling_strategy='all')
    X_res, y_res = cc.fit_resample(X_2, y_2)

    difficult_set_maj = X_res.copy(deep=True)
    difficult_set_maj['attack_type'] = y_res

    discrete = difficult_set_min.iloc[:,[1, 2, 3, 6, 11, 20, 21]]
    continuous = difficult_set_min.drop(columns=['Protocol_type','Service','Flag','Land','Logged_in','Is_hot_login','Is_guest_login','attack_type'])

    y_3 = difficult_set_min['attack_type']

    new_dfs = []
    number = int(difficult_set_maj.shape[0]/difficult_set_min.shape[0])
    # print(number)
    for n in range(50, 50+25):
        XD1 = discrete.copy(deep=True)
        XC1 = continuous * (1 - 1/n)
        XD2 = discrete.copy(deep=True)
        XC2 = continuous * (1 + 1/n)
        df1 = pd.concat([XD1, XC1, y_3], axis=1)
        df2 = pd.concat([XD2, XC2, y_3], axis=1)
        new_dfs.append(df1)
        new_dfs.append(df2)

    new_data = pd.concat(new_dfs, ignore_index=True)

    new_train = pd.concat([easy_set, difficult_set_maj, difficult_set_min, new_data],ignore_index=True)

    X_train_dsste = new_train.drop(columns=['attack_type'])
    y_train_dsste = new_train['attack_type']

    return X_train_dsste, y_train_dsste