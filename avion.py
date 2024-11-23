import pandas as pd
import matplotlib.pyplot as plt

"""**Year**  -  godina leta

**Month**  -  mesec leta

**DayOfMonth**  -  dan leta

**DayOfWeek**  -  kog dana u nedelji se desio let (1 - ponedeljak, 2 - utorak itd.)

**ElapsedTime**  -  koliko minuta je ukupno trajao let

**AirTime**  -  koliko minuta je avion bio u vazduhu

**ArrivalDelay**  -  koliko je kasnio avion u dolasku (negativna vrednost znači da je sleteo pre očekivanog vremena)

**DepartureDelay**  -  koliko je kasnio avion u polasku

**Origin**  -  skraćenica polaznog grada (odakle je krenuo avion)

**Destination**  -  skraćenica odredišta

**Distance**  - rastojanje između gradova u km

"""

"""# 1. Priprema skupa podataka"""

data = pd.read_csv('./DelayedFlights.csv', index_col=0)
#print(data)

"""Nedostajuci podaci"""

data[data.isna().any(axis=1)]

data = data[~data['ArrivalDelay'].isna()].reindex()
raw = data.copy() #STARA TABELA SA IZBACENIM NONE VRIJEDNOSTIMA

#data.head(4)

#raw.head(4)

"""Uklonila sam sve redove gdje mi je `ArrivalDelay` None ,obzirom da bez arrival delaya nista ne mozemo. Srecom, ne postoje redovi gdje `ArrivalDelay` dobar,a ostale kolone ne.

## Obrada podataka

Zanimaju me:

a) dan u godini

b) dan u sedmici

"""

import datetime
import numpy as np

"DAN U GODINI"
def day_of_year(row):
    year, month, day = row['Year'],	row['Month'],	row['DayofMonth']
    date_object = datetime.datetime(year, month, day)
    day_of_year = date_object.timetuple().tm_yday #vraca day of year
    total = (datetime.date(year, 12, 31) - datetime.date(year, 1, 1)).days + 1 #ukupan broj ana u godini
    return day_of_year, total
  
def day_of_year_sin(row):
    d, total = day_of_year(row)
    return np.sin(2 * np.pi * d / total)
  
def day_of_year_cos(row):
    d, total = day_of_year(row)
    return np.cos(2 * np.pi * d / total)

data['year_day_sin'] = data.apply(day_of_year_sin, axis=1)
data['year_day_cos'] = data.apply(day_of_year_cos, axis=1)
#print(data)

"DAN U SEDMICI"
data = pd.get_dummies(data, columns=['DayOfWeek'], drop_first=True, dtype=float)
#pretvara kolo

"IMENA GRADOVA U BROJEVE"

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Origin'] = le.fit_transform(data['Origin'])
data['Destination'] = le.fit_transform(data['Destination'])

del data['Month']
del data['DayofMonth']
#print(data)

data = data.astype('float64')
#data.info()

"""# 2. Eksplorativna analiza podataka

## GRADOVI I AERODROMI NA KARTI (VAN ZADATKA)
"""

iso = pd.read_csv('./airports.csv', index_col=0, names=['name', '2', '3', 'iso3', '5', 'lat', 'lng', '8', '9', '10', '11', '12', '13'])
iso.head(1)

iso[iso['iso3'] == 'IND']

avg_arrival_delay = raw.groupby('Origin')['ArrivalDelay'].sum() / raw.groupby('Origin')['ArrivalDelay'].count()
avg_arrival_delay = avg_arrival_delay.reset_index()
merged_df = iso[['iso3', 'lat', 'lng']].merge(avg_arrival_delay, left_on='iso3', right_on='Origin')
merged_df['Total'] = raw.groupby('Origin')['ArrivalDelay'].count().reset_index()['ArrivalDelay']

merged_df['ArrivalDelay'] = merged_df['ArrivalDelay'] / merged_df['ArrivalDelay'].max()
merged_df['Total'] = merged_df['Total'] / merged_df['Total'].max() * 200 + 10

merged_df

import geopandas

cmap = plt.cm.get_cmap('RdYlGn')
fig, ax = plt.subplots(figsize=(15, 12))
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world = world[world['iso_a3'] == 'USA']
world.plot(ax=ax)
ax.scatter(merged_df.lng, merged_df.lat, zorder=1, c=merged_df['ArrivalDelay'], s=merged_df['Total'], cmap=cmap)
plt.show()

"""Cikago radi bolje nego Kalifornija.
Zapadna obala je najgora

###########################################

### Zavisnost dana u nedjelji i kasnjenja leta
"""

avg = raw.groupby('DayOfWeek')['ArrivalDelay'].mean()
x, y = avg.index, avg.values

plt.title('Delay from Day Of Week')
plt.xlabel('Day')
plt.ylabel('Delay')
plt.bar(x, y)
plt.show()

groups = []
for i in range(1, 8):
  groups.append(raw[raw['DayOfWeek'] == i]['ArrivalDelay'].values)
#boxplot
plt.title('Delay from Day Of Week')
plt.xlabel('Day')
plt.ylabel('Delay')
plt.boxplot(groups)
plt.ylim(-100, 1200)
plt.show()

"""Malko veca cekanja utorkom i petkom

### Zavisnost udaljenost izmedju gradova i kasnjenja
"""

avg = raw.sort_values(by=['Distance']).groupby('Distance')['ArrivalDelay'].mean()
x, y = avg.index, avg.values

plt.title('Delay from distance')
plt.xlabel('Dist')
plt.ylabel('Delay')
plt.plot(x, y)
plt.show()

#scatter
plt.title('Delay from distance')
plt.xlabel('Dist')
plt.ylabel('Delay')
plt.scatter(raw['Distance'], raw['ArrivalDelay'])
plt.show()

groups = []
for a, b in [(i * 1000, i * 1000 + 1000) for i in range(0, 4)]:
  groups.append(raw[raw['Distance'].between(a, b)]['ArrivalDelay'].values)
#box plot
plt.title('Delay from distance')
plt.xlabel('Dist')
plt.ylabel('Delay')
plt.boxplot(groups)
plt.ylim(-100, 300)
plt.show()

"""Postoji vise kratkih putovanja, sto rezultira vecim kasnjenjima na kratkim putovanjima, ali sve ukupno, nema neke jasne razlike izmedju kratkih i dugih putovanja

## Meni je najintuitivnije kasnjenje u zavosnosti od Departure delay
"""

plt.title('Delay from delay :)')
plt.xlabel('Dep. Delay')
plt.ylabel('Arr. Delay')
plt.scatter(raw['DepartureDelay'], raw['ArrivalDelay'])
plt.show()

"""

# 3. MODELI MASINSKOG UCENJA 

## Podjela na Train\Test i Batches zbog ogranicene memorije
"""

from sklearn.model_selection import train_test_split

#DA LI KASNI? 1 da 0 ne
y = (data['ArrivalDelay'] > 0).astype('int')
del data['ArrivalDelay']

y.value_counts()

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

batches = []
for i in range(0, len(X_train), 10000):
  batches.append((X_train[i:i+10000],y_train[i:i+10000]))

"""## Koji model koristiti?

Vecina jednostavnih modela ima mogucnost racunanja feature importance:
- Logisticka regresija
- SVM (Support Vector Machine) - velika slozenost, pa necemo

Mozemo koristiti i stabla:
- Decision tree - Stablo odlucivanja
- Decision forest - Ansambl stabla odlucivanja


## LOGISTICKA REGRESIJA

"""

model = LogisticRegression(warm_start =True)

model.fit(X_train, y_train)
print(f"Train acc:\t{accuracy_score(y_train, model.predict(X_train)):.6f}")
print(f"Test acc:\t{accuracy_score(y_test, model.predict(X_test)):.6f}")

"""Nema overfittinga"""

coefficients = model.coef_[0]

feature_importance = pd.DataFrame({'Feature': data.columns, 'Importance': np.abs(coefficients)})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))

feature_importance = pd.DataFrame({'Feature': data.columns, 'Importance': coefficients})
print(feature_importance)

"""
## Decision Tree
"""

from sklearn.datasets import load_iris
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
print(f"Train acc:\t{accuracy_score(y_train, clf.predict(X_train)):.6f}")
print(f"Test acc:\t{accuracy_score(y_test, clf.predict(X_test)):.6f}")

"""**Overfitting!!!!**"""

clf.get_depth()
#ponovo testiram model sa ogranicenom dubinom
clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(X_train, y_train)
print(f"Train acc:\t{accuracy_score(y_train, clf.predict(X_train)):.6f}")
print(f"Test acc:\t{accuracy_score(y_test, clf.predict(X_test)):.6f}")

tree.plot_tree(clf)

""""""

for i in [1, 2, 3, 4, 8, *list(range(12, 20)), 25, 30, 38]:
  clf = tree.DecisionTreeClassifier(max_depth=i)
  clf = clf.fit(X_train, y_train)
  print(f"Train acc [max depth={i}]:\t{accuracy_score(y_train, clf.predict(X_train)):.6f}", end='\t\t')
  print(f"Test acc [max depth={i}]:\t{accuracy_score(y_test, clf.predict(X_test)):.6f}")

"""0.915697 (log reg) vs 0.920926 (dtree na dubini 15, tu je najbolja)

"""
"MEAN DECREASE IN IMPURITY- feature importance na osnovu smanjenja impuriteta, sto je mjera nesortiranosti ili necistoce u stablu odlucivanja"
"vece smanjenje impuriteta, vaznija karakteristika"
clf = tree.DecisionTreeClassifier(max_depth=15)
clf = clf.fit(X_train, y_train)
importances = clf.feature_importances_
std = np.std([clf.feature_importances_], axis=0)

forest_importances = pd.Series(importances, index=data.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

"""## RANDOM FOREST"""

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)
print(f"Train acc:\t{accuracy_score(y_train, clf.predict(X_train)):.6f}")
print(f"Test acc:\t{accuracy_score(y_test, clf.predict(X_test)):.6f}")

np.array([[estimator.get_depth() for estimator in clf.estimators_]])

"""Desava se overfitting, da bismo to uklonili, mozemo probati da dodamo jos stabala"""

clf = RandomForestClassifier(max_depth=35)
clf = clf.fit(X_train, y_train)
print(f"Train acc [max depth={i}]:\t{accuracy_score(y_train, clf.predict(X_train)):.6f}", end='\t\t')
print(f"Test acc [max depth={i}]:\t{accuracy_score(y_test, clf.predict(X_test)):.6f}")

"Train acc [max depth=35]:	0.999985		Test acc [max depth=35]:	0.932614"

clf = RandomForestClassifier(max_depth=30)
clf = clf.fit(X_train, y_train)
print(f"Train acc [max depth={i}]:\t{accuracy_score(y_train, clf.predict(X_train)):.6f}", end='\t\t')
print(f"Test acc [max depth={i}]:\t{accuracy_score(y_test, clf.predict(X_test)):.6f}")

"Train acc [max depth=30]:	0.999982		Test acc [max depth=30]:	0.935894"

"""NAJBOLJI REZULTAT"""

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

forest_importances = pd.Series(importances, index=data.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

