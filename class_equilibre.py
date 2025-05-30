import os
import matplotlib.pyplot as plt

def equilibre_classe(dataset_dir):
# 📊 Comptage des images par dossier (chaque dossier = une classe)
    class_counts = {}
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            count = len([
                fname for fname in os.listdir(class_path)
                if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            class_counts[class_name] = count

    # 📈 Affichage de l'histogramme
    plt.figure(figsize=(8, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color=['skyblue', 'salmon'])
    plt.title("Fréquence des images par classe")
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'images")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
# 📁 Chemin de ton dossier dataset
dataset = r'C:\Users\asus\Downloads\architectural-styles-dataset'  # Modifie si besoin
equilibre_classe(dataset)


import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# Chemin du dossier "moderne"
input_dir = r"C:\Users\asus\Downloads\architectural-styles-dataset/moderne"
output_dir =input_dir  # On enregistre les images directement dans le même dossier

# Nombre d'images existantes
current_images = os.listdir(input_dir)
current_count = len(current_images)
target_count = len(os.listdir(r'C:\Users\asus\Downloads\architectural-styles-dataset\hystorique'))  # Objectif

# Création du générateur d'images
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'   # Lorsqu’une image est transformée (décalée, tournée, etc.), certains pixels en bordure n’ont pas de valeur,remplit ces pixels vides en copiant les pixels les plus proches.
)

# Augmenter jusqu'à atteindre target_count
generated = 0
i = 0

while current_count + generated < target_count:
    image_name = current_images[i % len(current_images)]
    img_path = os.path.join(input_dir, image_name)

    # Charger et convertir l'image
    img = load_img(img_path)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Générer une seule image augmentée
    aug_iter = datagen.flow(x, batch_size=1)
    aug_img = next(aug_iter)[0].astype('uint8')

    # Sauvegarder l'image augmentée
    new_filename = f"aug_{generated}_{image_name}"
    save_path = os.path.join(output_dir, new_filename)
    array_to_img(aug_img).save(save_path)

    generated += 1
    i += 1

print(f"✅ {generated} images générées. Total: {current_count + generated}")
print("apres augmantation de class moderne")
equilibre_classe(dataset)