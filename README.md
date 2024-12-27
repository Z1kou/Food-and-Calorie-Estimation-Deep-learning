
Food and Calorie Estimation Using Deep Learning

📋 Description

Ce projet utilise des modèles de deep learning pour effectuer la classification d’aliments à partir d’images et fournir une estimation approximative des calories. Il s’appuie sur le dataset ECUSTFD et inclut des étapes de prétraitement, d’entraînement, et d’évaluation des performances.

🛠️ Fonctionnalités
	•	Classification d’images alimentaires : Identification de différentes catégories d’aliments.
	•	Estimation des calories : Calcul approximatif des calories en fonction de la classe d’aliment.
	•	Organisation automatisée des datasets : Scripts pour structurer les données en ensembles d’entraînement et de test.
	•	Modèle basé sur ResNet18 : Fine-tuning d’un modèle préentraîné pour une classification efficace.

📂 Structure du projet

Food-and-Calorie-Estimation-Deep-learning/
├── Data/                     # Scripts et datasets de base
├── Models/                   # Modèles implémentés
├── Training code/            # Scripts pour l'entraînement
├── dataset/                  # Organisation des images d'entraînement et de test
├── organisation.py           # Script pour organiser les datasets
├── classification_model.py   # Script principal pour l'entraînement et l'évaluation
├── README.md                 # Documentation

📦 Installation
	1.	Clonez ce dépôt :

git clone https://github.com/<username>/Food-and-Calorie-Estimation-Deep-learning.git
cd Food-and-Calorie-Estimation-Deep-learning


	2.	Créez un environnement virtuel :

python -m venv env
source env/bin/activate  # Sur Windows : env\Scripts\activate


	3.	Installez les dépendances :

pip install -r requirements.txt

🚀 Utilisation

Organisation des données

Exécutez le script pour organiser le dataset en ensembles train et test :

python organisation.py

Entraînement du modèle

Lancez le script principal pour entraîner et évaluer le modèle :

python classification_model.py

📊 Résultats
	•	Précision du modèle : ~69.61% sur le dataset ECUSTFD.
	•	Modèle utilisé : ResNet18 (Fine-tuning avec ImageNet).

📚 Dataset

Le projet utilise le dataset ECUSTFD. Téléchargez et placez-le dans le dossier dataset/ avant de lancer le script.

🌟 Prochaines Améliorations
	•	Amélioration de la précision avec des modèles plus avancés (ResNet50, EfficientNet, etc.).
	•	Augmentation de données et techniques d’enrichissement.
	•	Ajout d’une interface utilisateur pour une utilisation interactive.

🤝 Contributions

Les contributions sont les bienvenues ! Créez une issue ou ouvrez une pull request pour discuter des changements proposés.

⚖️ Licence

Ce projet est sous licence MIT. Consultez le fichier LICENSE pour plus de détails.
