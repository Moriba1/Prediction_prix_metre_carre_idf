import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import haversine_distances
from sklearn.model_selection import GridSearchCV

#I) Nettoyage des données
# importation des données 
dataset = pd.read_csv("full.csv", low_memory=False)
PARIS = np.radians([[48.8566, 2.3522]]) #les coor de Paris (latitude,longitude)
# Vérifiaction rapide des données et observation de la "forme" des données
print(dataset.head())
print(dataset.shape)

#Affichage de toutes les colonnes (=attributes)
#Dans notre cas, il y en a clairement des "plus intéréssantes"
#for i in dataset :
#   print(i)


#On va préparer pour demain, les conditions pour filtrer le datasets : 
# Voir carnet de bord


dep = ["91","92","93","94","95","75","78","77"]
dataset = dataset[dataset["code_departement"].isin(dep)] 

#On pourra remarquer un changement drastique !
print("Après choix ile de France : ",dataset.shape)

#inutile ducoup
dataset = dataset[(dataset["surface_reelle_bati"] >= 1 ) &
(dataset["surface_reelle_bati"]<=800)]

dataset = dataset[dataset["type_local"].isin(["Maison", "Appartement"])]
#Ici, aussi !
print("Après choix Maison | Appartement: ",dataset.shape)

important_rows = ["valeur_fonciere","surface_reelle_bati","longitude","latitude"]

dataset = dataset.dropna(subset=important_rows)





#Méthode IQR
qua1 = dataset["prix_metre2"].quantile(0.25)
qua3 = dataset["prix_metre2"].quantile(0.75)

IQR = qua3 - qua1
lower = qua1 - 1.5 * IQR
upper = qua3 + 2 * IQR

#Je pesnais pas que c'était nécéssaire mais permet peut-etre d'éviter les inf
dataset = dataset[dataset["nombre_pieces_principales"] > 0]

dataset = dataset[(dataset["prix_metre2"] >= lower) &
(dataset["prix_metre2"]<=upper)]


print("Dataset final !",dataset.shape)
# Ajout(s)
dataset = dataset[(dataset["surface_reelle_bati"] != 0)]
dataset = dataset.drop_duplicates()

print("Dataset final !",dataset.shape)


#Jour 3
#voir log / folium
#
def g() : 
    plt.hist(dataset["prix_metre2"], bins=100)
    plt.title("Distribution du prix au m2 - IDF")
    plt.xlabel('Prix au m2')
    plt.ylabel('Fréquence')
    plt.savefig('graphique.png')

def g2():
    plt.hist(dataset["surface_reelle_bati"], bins=100)
    plt.title("Répartition des surfaces")
    plt.xlabel('Surface en  m2')
    plt.ylabel('Fréquence')
    plt.savefig('graphique2.png')

def g3() : 
    plt.hist(dataset["nombre_pieces_principales"], bins=100)
    plt.title("Répartition des pièces")
    plt.xlabel('Surface en  m2')
    plt.ylabel('Fréquence')
    plt.savefig('graphique3.png')

def g4() :
    t = dataset.groupby("code_departement")["prix_metre2"].mean().reset_index()
    print(t)
    x = t["code_departement"]
    y = t["prix_metre2"]
    plt.plot(x,y)
    plt.savefig('graphique4.png')

def g5():
    tprime = dataset.groupby("type_local")["prix_metre2"].mean().reset_index()
    x = t["type_local"]
    y = t["prix_metre2"]
    plt.plot(x,y)
    plt.savefig('graphique5.png')



#Voir comment intérpréter le resultat ! 

#Deniers nettoyage !
print(dataset.shape)
print(dataset.isna().sum())

#II) Ajout des features
dataset = pd.get_dummies(dataset,columns=["type_local"], dtype = int)

dataset = pd.get_dummies(dataset,columns=["code_departement"], dtype = int, drop_first=True)

dataset["prix_metre2"] = dataset["valeur_fonciere"] / dataset["surface_reelle_bati"]
dataset["surface_piece"] = dataset["surface_reelle_bati"] / dataset["nombre_pieces_principales"]
print(dataset.corr(numeric_only=True))
# C'est l'un des derniers ajouts !
prix_m2_commune = dataset.groupby("nom_commune")["prix_metre2"].mean()
dataset["prix_m2_commune"] = dataset["nom_commune"].map(prix_m2_commune)
#On suppose que la date = JJ/MM/AAAA
dataset["annee"] = pd.to_datetime(dataset["date_mutation"]).dt.year


coords = np.radians(dataset[["latitude", "longitude"]])
distance = haversine_distances(coords, PARIS)
dataset["distance_paris"] = distance * 6371

attributs = [
"surface_reelle_bati",
"nombre_pieces_principales",
"surface_piece",
"type_local_Maison",
"code_departement_77",
"code_departement_78",
"code_departement_91",
"code_departement_92",
"code_departement_93",
"code_departement_94",
"code_departement_95",
"prix_m2_commune",
"annee",
"longitude",
"latitude",
"distance_paris"
]

X = dataset[attributs]
y = dataset["prix_metre2"]

#III) Entrainement du modèle
# Apprentissage & Partitionnement
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_test = y_test.astype(float)
y_train = y_train.astype(float)




pipeline_v1 = make_pipeline(
    LinearRegression()
)
pipeline_v2 = make_pipeline(
    StandardScaler(),
    LinearRegression()
)
pipeline = make_pipeline(
    RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
)
)
pipeline_v4 = make_pipeline(
    StandardScaler(),
    MinMaxScaler(),
    LinearRegression()
)
#Optimisation Random Froest
#puis predict puis score voilà voilà


# Pour la conclusion : 
def evaluate (name, model, X_train, y_train, X_test, y_test) :
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return name,score,rmse


compare = []
compare.append(evaluate("Linear Regression ",pipeline_v1, X_train, y_train, X_test, y_test))
compare.append(evaluate("SS Linear Regression ",pipeline_v2, X_train, y_train, X_test, y_test))
compare.append(evaluate("Random Forest ",pipeline, X_train, y_train, X_test, y_test))
compare.append(evaluate("MM SS Linear Regression ",pipeline_v4, X_train, y_train, X_test, y_test))

temp = pd.DataFrame(compare)
print(temp)


dataset.to_csv("clean_dataset.csv", index = True)

