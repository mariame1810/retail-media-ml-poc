# Retail Media – Machine Learning POC

## Contexte
Ce projet est un Proof of Concept inspiré des problématiques de retail media et de ciblage promotionnel.
L’objectif est de montrer comment des données transactionnelles (tickets de caisse) peuvent être transformées
en décisions de ciblage marketing mesurables à l’aide du machine learning.

## Objectif
Prédire la probabilité qu’un client rachète un produit afin de prioriser les clients à cibler lors d’une campagne promotionnelle,
et mesurer l’impact business potentiel de ce ciblage.

## Données
Le projet utilise le dataset public **Instacart Market Basket Analysis**, composé de :
- commandes clients
- historique d’achats produits
- informations temporelles (récence, fréquence)

Ces données sont représentatives de tickets de caisse anonymisés.

## Approche
1. Exploration des données et validation rapide du POC dans un notebook.
2. Création de features client (récence, fréquence).
3. Construction d’un dataset de machine learning avec exemples positifs et négatifs.
4. Entraînement d’un modèle de propension au rachat (régression logistique).
5. Simulation d’une campagne promotionnelle basée sur le score du modèle.

Le code clé a été refactorisé dans des modules afin de séparer :
- la création des features
- la construction du dataset
- l’entraînement et l’évaluation du modèle

## Résultats
- AUC du modèle : **0.73**
- En ciblant les 10 % de clients les mieux scorés :
  - taux de rachat non ciblés : ~47 %
  - taux de rachat ciblés : ~79 %
  - gain de plus de **30 points de taux de rachat**

Ces résultats montrent que le modèle est directement activable dans un contexte de retail media.

## Structure du projet
luckycart-retail-ml/
├── data/
├── notebooks/
│ └── 01_exploration.ipynb
├── src/
│ ├── features/
│ ├── data/
│ └── model/
└── README.md
## Prochaines étapes
- Enrichissement des features (produits, catégories, saisonnalité).
- Validation via A/B testing pour mesurer l’incrémentalité réelle.
- Mise en production dans un pipeline batch ou temps réel avec monitoring.
