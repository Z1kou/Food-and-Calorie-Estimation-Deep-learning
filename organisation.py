import os
from shutil import copy

# Chemins des fichiers
images_dir = "dataset/JPEGImages"  # Dossier contenant toutes les images
annotations_dir = "dataset/ImageSets/Main"  # Dossier contenant les fichiers train.txt et test.txt
output_dir = "dataset"  # Dossier où seront créés `train` et `test`

# Classes du dataset ECUSTFD
classes = [
    "Apple", "Banana", "Bread", "Bun", "Doughnut", "Egg", "Fried Dough Twist", "Grape",
    "Lemon", "Litchi", "Mango", "Mooncake", "Orange", "Peach", "Pear", "Plum",
    "Kiwi", "Sachima", "Tomato"
]

# Fonction pour lire les fichiers train/test
def read_file(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

# Créer les dossiers de sortie par classe
for split in ["train", "test"]:
    for class_name in classes:
        os.makedirs(f"{output_dir}/{split}/{class_name}", exist_ok=True)

# Lire les fichiers train et test
train_files = read_file(os.path.join(annotations_dir, "train.txt"))
test_files = read_file(os.path.join(annotations_dir, "test.txt"))

# Fonction pour associer un fichier d'image à une classe
def get_class_name(filename):
    for class_name in classes:
        if class_name.lower() in filename.lower().replace("_", " "):  # Vérifie si le nom de la classe est dans le nom de fichier
            return class_name
    return None  # Si aucune classe n'est trouvée

# Copier les images dans les dossiers respectifs
for img in train_files:
    class_name = get_class_name(img)
    if class_name:
        img_path = os.path.join(images_dir, img + ".jpg")  # Modifie l'extension si nécessaire
        if os.path.exists(img_path):
            copy(img_path, f"{output_dir}/train/{class_name}")

for img in test_files:
    class_name = get_class_name(img)
    if class_name:
        img_path = os.path.join(images_dir, img + ".jpg")  # Modifie l'extension si nécessaire
        if os.path.exists(img_path):
            copy(img_path, f"{output_dir}/test/{class_name}")

print("Les images ont été organisées avec succès.")