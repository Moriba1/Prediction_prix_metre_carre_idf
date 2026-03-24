Prédiction du prix au m2 en Île-de-France

Objectif

Ce projet a pour objectif de prédire le prix au mètre carré de biens immobiliers en Île-de-France à partir de données publiques (DVF).
L'objectif est d'explorer différents modèles de machine learning afin d'identifier celui offrant les meilleures performances.

Données du devoir

Les données utilisées proviennent des transactions immobilières (DVF).
Après nettoyage, le dataset a été restreint aux départements d’Île-de-France et aux biens de type maison et appartement et d'autres contraintes sur la surface notamment.

Principales variables utilisées :

surface réelle bâtie
nombre de pièces principales
localisation (latitude, longitude)
distance à Paris
prix moyen par commune
type de bien (Maison, Appartement)
année
code département 


Méthodologie

- Nettoyage des données
suppression des valeurs manquantes (notamment latitude et longitude)
filtrage des surfaces incohérentes
suppression des données hors-normes ou outliers (méthode IQR)
suppression des doublons
- Feature engineering
création du prix au m2
calcul de la distance à Paris (formule de Haversine)
ajout du prix moyen par commune
création de variables dérivées (surface par pièce)
- Encodage
transformation des variables catégorielles (one-hot encoding), tels que le département par exemple
- Modèles testés
Régression linéaire
Régression linéaire avec standardisation
Régression linéaire avec MinMaxScaler
Random Forest
- Résultats
Modèle			R²	RMSE
Linear Regression	0,55	2477,24
Scaled Linear Regression 0,55	2477,23
Random Forest		0,63	2242,42

Le modèle Random Forest obtient les meilleures performances.

Limites
certaines variables importantes sont absentes (état du bien, étage, proximité à des zones fortes etc.)
possible biais lié au découpage train/test
données géographiques encore simplifiées
Améliorations possibles
ajout de nouvelles features (quartier, transports, etc.)
optimisation des hyperparamètres (GridSearch)
validation croisée
test de modèles plus avancés


Auteur :
Soukouna Moriba
