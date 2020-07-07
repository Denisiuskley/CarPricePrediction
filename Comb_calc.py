try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import  pandas as pd
import numpy as np 
from scipy.optimize import curve_fit
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from catboost import CatBoostRegressor, Pool
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from tqdm import tqdm
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

def percentil5 (x):
    x = x.dropna()
    if x.shape[0] > 0:
        return np.round(np.percentile(x, q=5), 3)
    else: return 0
def percentil95 (x): 
    x = x.dropna()
    if x.shape[0] > 0:
        return np.round(np.percentile(x, q=95), 3)
    else: return 0
    
def group_stat(df, group, for_stat):
    gr = df.groupby(group).agg(
        Par_min = (for_stat, 'min'),
        Par_quantil1 = (for_stat, percentil5),
        Par_median = (for_stat, 'median'),
        Par_mean = (for_stat, 'mean'),
        Par_quantil3 = (for_stat, percentil95),
        Par_max = (for_stat, 'max'),
        Par_std = (for_stat, 'std'),
        Par_count = (for_stat, 'count')).reset_index()
    return gr

def plot_silh(X, cluster_labels, n_clusters):
    from sklearn.metrics import silhouette_score, silhouette_samples
    import matplotlib.cm as cm
    
    silhouette_avg = silhouette_score(X=X, labels=cluster_labels)
    sample_silhouette_values = silhouette_samples(X=X, labels=cluster_labels)
        
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 10)
    
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
    
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
    
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    # 2nd Plot showing the actual clusters formed
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

def compute_meta_feature(model, X_train, X_test, y_train, cv, cat_features):
    X_meta_train = np.zeros((len(y_train), 1), dtype=np.float32)
    splits = cv.split(X_train)
    for train_fold_index, predict_fold_index in splits:
        X_fold_train, X_fold_predict = X_train[train_fold_index], X_train[predict_fold_index]
        y_fold_train = y_train[train_fold_index]
        
        folded_clf = clone(model)
        folded_clf.fit(X_fold_train, y_fold_train, cat_features = cat_features)
        
        X_meta_train[predict_fold_index] = folded_clf.predict(X_fold_predict).reshape(-1, 1)
    
    meta_clf = clone(model)
    meta_clf.fit(X_train, y_train, cat_features = cat_features)
    
    X_meta_test = meta_clf.predict(X_test).reshape(-1, 1)
    return X_meta_train, X_meta_test

def generate_meta_features(models, X_train, X_test, y_train, cv, cat_features):
   
    features = [
        compute_meta_feature(model, X_train, X_test, y_train, cv, cat_features)
        for model in tqdm(models)
    ]
    stacked_features_train = np.hstack([
        features_train for features_train, features_test in features
    ])

    stacked_features_test = np.hstack([
        features_test for features_train, features_test in features
    ])
    
    return stacked_features_train, stacked_features_test

def compute_metric(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def func(x, a, b):     
    return a * x + b

#data = pd.DataFrame()
#
#for i in range(1, 200):
#    filename = f'Parcing/my_parce_{i}.csv'
#    temp = pd.read_csv(filename)
#    data = data.append(temp, sort=False)
#
#data = data.dropna(subset = ['bodyType'])
#
#data.to_csv('train.csv', index = False)

df = pd.read_csv('train.csv')
df = df.drop_duplicates()
dft = pd.read_csv('test.csv')

cols = list(df.columns)

gr = group_stat(dft, 'bodyType', 'numberOfDoors')
types = list(gr['bodyType'])

df = df.loc[df['bodyType'].isin(types), :]

df['numberOfDoors'] = df['bodyType']
d = gr[['bodyType', 'Par_mean']].to_dict('split')
dd = dict(d['data'])

df['engineDisplacement'] = df['fuelType']
df['enginePower'] = df['fuelType']
df['fuelType'] = df.apply(lambda x: re.findall(r'\w+', x.fuelType)[-1].lower(), axis = 1)

df['engineDisplacement'] = df.apply(lambda x: 0 if x.fuelType == 'электро' else float(x.engineDisplacement[:3]), axis = 1)
df['enginePower'] = df.apply(lambda x: int(x.enginePower[:3]) if x.fuelType == 'электро' else int(re.search(r'\/(.*?)\/', x.enginePower).group(1)[:-5]), axis = 1)

df['Состояние'] = df['Состояние'].apply(lambda x: x if x == 'Не требует ремонта' else np.nan)
df = df.dropna(subset = ['Состояние', 'Price'])
cleanup_nums = {'numberOfDoors': dd}
df.replace(cleanup_nums, inplace=True)
df['numberOfDoors'] = df['numberOfDoors'].astype(int)
df = df.loc[df['Владельцы'] != 'Оригинал']
df = df.loc[df['ПТС'] != 'Растаможен']

dft['engineDisplacement'] = dft['engineDisplacement'].apply(lambda x: float(x[:3]) if x[:3] != 'und' else 0)
dft['enginePower'] = dft['enginePower'].apply(lambda x: int(x[:3]))

dft['Price'] = np.nan

'''Преобразуем год выпуска к возрасту'''
dft['years'] = 2021 - dft['productionDate']
df['years'] = 2021 - df['productionDate']

'''Удаляем раритеты, которых нет в тестовом наборе'''
max_years = dft.loc[dft['Price'].isnull(), 'years'].max()
dft = dft[dft['years'] <= max_years]

'''Если есть авто с похожими характеристиками, то просто возьмем среднее, 
для остальных - результаты моделирования'''
dftd = pd.merge(dft, df, on=['name', 'bodyType', 'fuelType', 'productionDate',
                             'engineDisplacement', 'enginePower', 'mileage'], how='inner')
dftd = dftd[['id_x', 'Price_y']]
dftd.columns = ['id','Price dup']
dftd = dftd.groupby('id')['Price dup'].agg(['median']).reset_index()

dropcol = ['brand', 'modelDate', 'vehicleConfiguration', 'Комплектация', 'Руль',
           'Таможня', 'Владение', 'Таможня', 'Состояние', 'id']
dft.drop(dropcol, axis = 1, inplace = True)
df.drop(dropcol, axis = 1, inplace = True)
dft = dft.append(df, sort=False).reset_index(drop=True)

'''Преобразуем в числовые величины'''
dft['Владельцы'] = dft['Владельцы'].apply(lambda x: int(x[:1]))
cleanup_nums = {'ПТС': {'Оригинал': 0, 'Дубликат': 1}}
dft.replace(cleanup_nums, inplace=True)

'''xDrive в имени модели появилось сравнительно недавно, поэтому удалим это 
из имени, но создадим параметр. Преобразуем имя.'''
dft['xDrive'] = dft['name']
dft['xDrive'] = dft['xDrive'].apply(lambda x: 1 if 'xDrive' in x else 0)
dft['name'] = dft['name'].apply(lambda x: x[:x.find(' ')])
dft['name'] = dft['name'].apply(lambda x: x[:-1] if x[-1] == 's' else x)
dft['name'] = dft['name'].apply(lambda x: x[:-1] if x[-1] == 'i' or x[-1] == 'd' else x)
dft['name'] = dft['name'].apply(lambda x: x[:-1] if x[-1] == 's' or x[-1] == 'x' or x[-1] == 'L' else x)
dft['name'] = dft['name'].apply(lambda x: x[6:] if x[:6] == 'sDrive' or x[:6] == 'xDrive' else x)
dft.loc[dft['name'] == '94Ah', 'name'] = 'Electro'
dft.loc[dft['name'] == 'ActiveHybri', 'name'] = 'Active'

'''Вытащим некоторые характерные слова из описания. Влияющие слова определял по 
величинам ошибок результатов обучения модели в конце.'''
dft['кожа'] = dft['description']
dft['кожа'] = dft['кожа'].apply(lambda x: 1 if 'кожа' in str(x).lower() else 0)
dft['фары'] = dft['description']
dft['фары'] = dft['фары'].apply(lambda x: 1 if 'светоди' in str(x).lower() or 'ксенон' in str(x).lower() else 0)
dft['кредит'] = dft['description']
dft['кредит'] = dft['кредит'].apply(lambda x: 1 if 'кредит' in str(x).lower() else 0)
dft['дтп'] = dft['description']
dft['дтп'] = dft['дтп'].apply(lambda x: 1 if 'дтп' in str(x).lower() or 'авари' in str(x).lower() else 0)
dft['идеал'] = dft['description']
dft['идеал'] = dft['идеал'].apply(lambda x: 1 if 'идеал' in str(x).lower() or 'отлич' in str(x).lower() else 0)
dft['launch'] = dft['description']
dft['launch'] = dft['launch'].apply(lambda x: 1 if 'launch' in str(x).lower() else 0)


'''Просматриваю статистику по параметрам, убеждаясь в значимости всех выбранных'''
gr = group_stat(dft, ['name'], 'Price')
names = list(gr['name'])
gr = group_stat(dft, ['name', 'years'], 'Price')

par = 'name'
gr = group_stat(dft, [par], 'Price')
gr2 = group_stat(dft, [par], 'years')

#dft['Price'] = np.log(dft['Price'])
#dft.sort_values(by = 'bodyType', inplace = True)
#dft['years discr'] = pd.cut(dft['years'], 5 , labels=False)
#fig, ax = plt.subplots(figsize=(10, 10))
#sns.boxplot(x="Price", y="bodyType", data=dft[dft['years discr'] == 2])
#
#fig, ax = plt.subplots(figsize=(10, 10))
#sns.boxplot(x="Price", y="fuelType", data=dft[dft['years discr'] == 2])
#
#fig, ax = plt.subplots(figsize=(10, 10))
#sns.boxplot(x="Price", y="color", data=dft[dft['years discr'] == 2])
#
#fig, ax = plt.subplots(figsize=(10, 10))
#sns.boxplot(x="Price", y="vehicleTransmission", data=dft[dft['years discr'] == 2])
#
#fig, ax = plt.subplots(figsize=(10, 10))
#sns.boxplot(x="Price", y="Привод", data=dft[dft['years discr'] == 2])
#
#fig, ax = plt.subplots(figsize=(10, 10))
#
#g = sns.relplot(x="years", y="mileage", hue = "fuelType", col="bodyType", size="Price", data=dft)
#g = sns.relplot(x="years", y="mileage", hue = "color", col="bodyType", size="Price", data=dft)
#g = sns.relplot(x="years", y="mileage", hue = "vehicleTransmission", col="bodyType", size="Price", data=dft)
#g = sns.relplot(x="years", y="mileage", hue = "Привод", col="bodyType", size="Price", data=dft)

#dft['Price'] = np.log(dft['Price'])

'''Комплексный параметр'''
dft['mil_y'] = np.log(dft['mileage'] * dft['years'])

fig, ax = plt.subplots(figsize=(10, 10))
sns.regplot(x="mil_y", y="Price", data=dft, x_estimator = np.median, x_bins = 100, ax = ax)


#sc = MinMaxScaler()
#j2 = 0
#i2 = 0
#dft['dpr'] = np.nan
#for name in names:    
#    data = dft.loc[dft['name'] == name, ['mil_y', 'Price']]
#    data.dropna(inplace = True)
#    if data.shape[0] > 3:
#        if i2 % 8 == 0:
#            fig, ax = plt.subplots(figsize=(40, 20), nrows = 2, ncols = 4)
#            j1 = 0
#            j2 = 0
#        if j2 % 4 == 0 and i2 % 8 != 0:
#            j1 = 1
#            j2 = 0
#        g = sns.regplot(data['mil_y'], data['Price'], scatter_kws={"s": 50}, ci = None, ax = ax[j1, j2])
#        ax[j1, j2].set_ylabel('Price')
#        ax[j1, j2].set_title(name)
#        ax[j1, j2].set_ylim([data['Price'].min()*0.95,data['Price'].max()*1.05])
#        data_sc = sc.fit_transform(data)
#        popt, pcov = curve_fit(func, data_sc[:,0], data_sc[:,1], maxfev = 100000)
#        y_pred = data_sc[:,1]- func(data_sc[:,0], *popt)
#        dft.loc[data.index, 'dpr'] = y_pred
#        i2 += 1
#        j2 += 1

'''Провожу кластеризацию по описанию. Хорошо выделяется 3 класса объявлений.'''
dft['description'] = dft['description'].apply(lambda x: 'автомобиль' if len(str(x)) < 5 else x)
dft['description'] = dft['description'].fillna('автомобиль')
dft['description'] = dft['description'].apply(lambda x: x.replace('.', '. '))
#dft['numb symb'] = dft['description']
#dft['numb symb'] = dft['numb symb'].apply(lambda x: len(x))

from sklearn.feature_extraction.text import CountVectorizer
analyzer = CountVectorizer().build_analyzer()
descr = []
for i in dft.index:
    descr.append(analyzer(dft.loc[i, 'description']))  

from gensim.models import Word2Vec
# Обучаем модель векторайзера на нашем наборе данных
# На выходе мы получим вектор признаков для каждого слова
n = 10
tsv = 2
model = Word2Vec(descr, min_count=10, size=n)
# Наивный подход к созданию единого эмбеддинга для документа – средний эмбеддинг по словам
def doc_vectorizer(doc, model):
    doc_vector = []
    num_words = 0
    for word in doc:
        if len(word) > 3:
            try:
                if num_words == 0:
                    doc_vector = model[word]
                else:
                    doc_vector = np.add(doc_vector, model[word])
                num_words += 1
            except:
                pass
    return np.asarray(doc_vector) / num_words

# Составляем эмбеддинги для наших документов
X = np.zeros((len(descr), n))
i2 = 0
for doc in descr:
    if len(doc) > 0:
        a = doc_vectorizer(doc, model)
        X[i2, 0 : a.shape[0]] = doc_vectorizer(doc, model)
    i2 += 1
sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn import decomposition
tsvd = decomposition.TruncatedSVD(tsv)
tsv_name = ['tsvd_' + str(i) for i in range(1, tsv + 1)]
df_ts = pd.DataFrame(tsvd.fit_transform(X), columns = tsv_name)
#dft = pd.concat([dft, df_ts], axis = 1)

from sklearn.mixture import GaussianMixture
n_clusters = 3
em_gm = GaussianMixture(n_components=n_clusters, 
                        max_iter=500,
                        init_params='kmeans')
em_gm.fit(df_ts)
cluster_labels = em_gm.predict(df_ts)
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x="tsvd_1", y="tsvd_2", data=df_ts, hue = cluster_labels, ax = ax)
plot_silh(df_ts, cluster_labels, n_clusters)

dft['cluster_labels'] = cluster_labels       


#cols = list(dft.columns)
#cols_d = ['engineDisplacement', 'enginePower', 'mileage', 'years', 'cluster_labels', 'numb symb']
#from scipy.stats import norm
#j2 = 0
#i2 = 0
#for par in cols_d:
#    if i2 % 8 == 0:
#        fig, ax = plt.subplots(figsize=(40, 20), nrows = 2, ncols = 4)
#        j1 = 0
#        j2 = 0
#    if j2 % 4 == 0 and i2 % 8 != 0:
#        j1 = 1
#        j2 = 0
#    sns.distplot(dft.loc[~dft['Price'].isnull(), par], hist = True, fit=norm, kde = False, ax=ax[j1, j2])
#    sns.distplot(dft.loc[dft['Price'].isnull(), par], hist = True, fit=norm, kde = False, ax=ax[j1, j2])
#
#    i2 += 1
#    j2 += 1
#for par in cols_d:
#    print(par, 'min', dft.loc[~dft['Price'].isnull(), par].min(), dft.loc[dft['Price'].isnull(), par].min())
#    print(par, 'max', dft.loc[~dft['Price'].isnull(), par].max(), dft.loc[dft['Price'].isnull(), par].max())
    


iskl = ['description', 'Price', 'productionDate']
df = dft.loc[~dft['Price'].isnull(), :]

X = df.drop(iskl, axis = 1)
y = df['Price']

list_cat_num = [0,1,2,3,5,9]
'''Предварительное обучение модели для поиска лучшего решения и причин больших ошибок'''
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

kf = KFold(n_splits=3, shuffle = True, random_state = 0)
feature_importances = pd.DataFrame()
feature_importances['feature'] = X.columns
i2 = 0
results = []
y_pred_m = np.zeros((y.shape[0]))
for train, test in kf.split(X, y):
    model = CatBoostRegressor(iterations = 5000,learning_rate = 0.02, depth = 7,
                                              l2_leaf_reg = 1, bagging_temperature = 1, 
                                              custom_metric = ['R2', 'MAE'], eval_metric = 'MAPE', verbose = 1000)
    model.fit(X.iloc[train,:], y.iloc[train], cat_features = list_cat_num)
    y_pred = model.predict(X.iloc[test,:])    
    y_pred_m[test] = y_pred
    feature_importances[i2] = model.get_feature_importance(data=None,
       prettified=False, thread_count=-1, verbose=False)
    print('RMSE score CatBoost:', np.round(mean_absolute_percentage_error(y.iloc[test], y_pred), 4))
    results.append(mean_absolute_percentage_error(y.iloc[test], y_pred))
    i2 += 1
print('CB:', np.round(np.mean(results), 5), np.round(metrics.r2_score(y, y_pred_m), 4))
fi_plot = pd.DataFrame()
for i in range(i2):
    temp = feature_importances[['feature',i]]
    temp.columns = ['feature', 'importance']    
    fi_plot = fi_plot.append(temp, sort=False)
feature_importances['mean'] = feature_importances.iloc[:,1:].mean(axis = 1)
fi_plot = pd.merge(fi_plot, feature_importances[['feature', 'mean']], 
                            on=['feature'], how='left')

ind = np.unravel_index(np.argsort(feature_importances['mean'], axis=None), feature_importances['mean'].shape)[0]

plt.figure(figsize=(16, 16))
sns.barplot(data=fi_plot.sort_values(by='mean', ascending=False), x='importance', y='feature', capsize=.2)

fig, ax = plt.subplots(figsize=(10, 10))
sns.regplot(y_pred_m, y, scatter_kws={"s": 40}, ax = ax)
raise Exception()

'''Вычисляем ошибки модели'''    
dft['price_pred'] = np.nan
dft.loc[y.index, 'price_pred'] = (y - y_pred_m) / y * 100

'''Оцениваем, что пишут неадекватные товарищи, завышающие или занижающие стоимость'''
word_price = dict()
wpl = []
df = dft.loc[~dft['price_pred'].isnull(), :]
for i in df.index:
    doc = re.findall(r'\w\w\w\w+', df.loc[i, 'description'].lower())
    for word in doc:
#        if word in word_price:
#            word_price[word] += df.loc[i, 'Price']
#        else:
#            word_price[word] = df.loc[i, 'Price']
        wpl.append([word, df.loc[i, 'price_pred']])
df_wp = pd.DataFrame(wpl, columns = ['word', 'price'])
gr = group_stat(df_wp, ['word'], 'price')
gr = gr[gr['Par_count'] > 5]
raise Exception

'''Также просматриваем на каких параметрах модель больше всего ошибается'''
#dft['mileage discr'] = pd.cut((dft['mileage'])**0.4, 9 , labels=False)
gr = group_stat(dft, ['name'], 'price_pred')
gr = gr[(gr['Par_count'] > 5) & (gr['Par_std'] > 20)]

gr = group_stat(dft, ['bodyType'], 'price_pred')
gr = gr[(gr['Par_count'] > 5) & (gr['Par_std'] > 20)]

par = 'years'
gr = group_stat(dft, [par], 'price_pred')
gr = gr[(gr['Par_count'] >= 3)]
fig, ax = plt.subplots(figsize=(20, 10), ncols = 2, nrows = 1)
sns.regplot(x=par, y="Par_std", data=gr, ax = ax[0])
sns.regplot(x=par, y="Par_median", data=gr, ax = ax[1])
for i in gr.index:
    ax[0].text(gr.loc[i, par], gr.loc[i, 'Par_std'], gr.loc[i, par])
    ax[1].text(gr.loc[i, par], gr.loc[i, 'Par_median'], gr.loc[i, par])

par = 'engineDisplacement'
gr = group_stat(dft, [par], 'price_pred')
gr = gr[(gr['Par_count'] >= 3)]
fig, ax = plt.subplots(figsize=(20, 10), ncols = 2, nrows = 1)
sns.regplot(x=par, y="Par_std", data=gr, ax = ax[0])
sns.regplot(x=par, y="Par_median", data=gr, ax = ax[1])
for i in gr.index:
    ax[0].text(gr.loc[i, par], gr.loc[i, 'Par_std'], gr.loc[i, par])
    ax[1].text(gr.loc[i, par], gr.loc[i, 'Par_median'], gr.loc[i, par])

par = 'enginePower'
gr = group_stat(dft, [par], 'price_pred')
gr = gr[(gr['Par_count'] >= 3)]
fig, ax = plt.subplots(figsize=(20, 10), ncols = 2, nrows = 1)
sns.regplot(x=par, y="Par_std", data=gr, ax = ax[0])
sns.regplot(x=par, y="Par_median", data=gr, ax = ax[1])
for i in gr.index:
    ax[0].text(gr.loc[i, par], gr.loc[i, 'Par_std'], gr.loc[i, par])
    ax[1].text(gr.loc[i, par], gr.loc[i, 'Par_median'], gr.loc[i, par])

fig, ax = plt.subplots(figsize=(10, 10))
sns.regplot(x="years", y="price_pred", data=dft, x_estimator = np.median, x_bins = 50, ax = ax)

fig, ax = plt.subplots(figsize=(10, 10))
sns.regplot(x="enginePower", y="price_pred", data=dft, x_estimator = np.median, x_bins = 50, ax = ax)

dft['years discr'] = pd.cut(dft['years'], 5 , labels=False)

fig, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x="price_pred", y="bodyType", data=dft[dft['years discr'] == 1])

fig, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x="Price", y="fuelType", data=dft[dft['years discr'] == 2])

fig, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x="Price", y="color", data=dft[dft['years discr'] == 2])

fig, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x="Price", y="vehicleTransmission", data=dft[dft['years discr'] == 2])

fig, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x="Price", y="Привод", data=dft[dft['years discr'] == 2])



'''Итоговое обучение модели. Стэкинг из Катбустов не дает прироста точности. 
Если заменять на другую модель, надо решать вопрос с декодированием переменных 
и использовать другой датафрейм. Можно пробовать при наличии времени'''
df = dft.loc[~dft['Price'].isnull(), :]
t_df = dft.loc[dft['Price'].isnull(), :]

y_train = df['Price']  
X_train = df.drop(iskl, axis = 1)
X_test = t_df.drop(iskl, axis = 1)

outputCB = pd.DataFrame({'id': X_test.index})

#models = [CatBoostRegressor(iterations = 4000,learning_rate = 0.02, depth = 8,
#                          l2_leaf_reg = 0.1, custom_metric = ['R2', 'MAE'], eval_metric = 'MAPE', verbose = 1000),
#          CatBoostRegressor(iterations = 6000,learning_rate = 0.02, depth = 7,
#                          l2_leaf_reg = 0.1, custom_metric = ['R2', 'MAE'], eval_metric = 'MAPE', verbose = 1000),
#          CatBoostRegressor(iterations = 10000,learning_rate = 0.02, depth = 6,
#                          l2_leaf_reg = 1, custom_metric = ['R2', 'MAE'], eval_metric = 'MAPE', verbose = 1000),
#          CatBoostRegressor(iterations = 10000,learning_rate = 0.02, depth = 7,
#                          l2_leaf_reg = 1, custom_metric = ['R2', 'MAE'], eval_metric = 'MAPE', verbose = 1000)]
#cv = KFold(n_splits=5, shuffle=True, random_state=42)
#stacked_features_train, stacked_features_test = generate_meta_features(models, X_train.values, X_test.values, y_train.values, cv, list_cat_num)
#clf = LinearRegression(normalize = True)
#y_pred = compute_metric(clf, stacked_features_train, y_train, stacked_features_test)
#outputCB['price'] = y_pred

for i in range(10):
    model = CatBoostRegressor(iterations = 50000,learning_rate = 0.02, depth = 7,
                                                  l2_leaf_reg = 1, bagging_temperature = 1, 
                                                  custom_metric = ['R2', 'MAE'], eval_metric = 'MAPE', verbose = 1000)
    
    X_train2, X_eval, y_train2, y_eval = train_test_split(X_train, y_train, test_size=0.3, random_state = i*10)
    eval_pool = Pool(X_eval, y_eval, cat_features = list_cat_num)
    model.fit(X_train2, y_train2, cat_features = list_cat_num, eval_set=eval_pool, early_stopping_rounds=200)
    y_pred = model.predict(X_test)
    outputCB[i] = y_pred
outputCB['price'] = outputCB.iloc[:,1:].mean(axis = 1)
'''5% инфляция'''
outputCB['price'] = outputCB['price'] / 1.05

output = pd.merge(outputCB, dftd, on=['id'], how='left')
out_f = output[['id', 'price', 'median']]
out_f.loc[out_f['median'] > 0, 'price'] = out_f.loc[out_f['median'] > 0, 'median']
out_f = out_f.drop(['median'], axis = 1)
out_f.columns = ['id', 'price']

#outputCB[['id','price']].to_csv('newpars_minus6_change_name.csv', index=False)
out_f[['id','price']].to_csv('minus5_with_dup.csv', index=False)


