import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil

# === 1. Organisation du dataset ===
original_data_dir = '/content/dataset/dataset'
base_dir = '/content/data_split'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

for split_dir in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(split_dir, 'modern'))
    os.makedirs(os.path.join(split_dir, 'historical'))

def split_dataset(class_name):
    path = os.path.join(original_data_dir, class_name)
    all_images = os.listdir(path)
    train_imgs, temp_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    for img in train_imgs:
        shutil.copy(os.path.join(path, img), os.path.join(train_dir, class_name))
    for img in val_imgs:
        shutil.copy(os.path.join(path, img), os.path.join(val_dir, class_name))
    for img in test_imgs:
        shutil.copy(os.path.join(path, img), os.path.join(test_dir, class_name))

split_dataset('modern')
split_dataset('historical')

# === 2. Prétraitement des données ===
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

val_generator = datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

test_generator = datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)

# === 3. Transfert Learning : MobileNetV2 base ===
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # On gèle toutes les couches du modèle

# === 4. Construction du modèle complet ===
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # binaire
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# === 5. Entraînement initial (seulement le top classifier) ===
early_stop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss')

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stop]
)

# === 6. Fine-Tuning ===
# On dégèle une partie des couches du modèle pré-entraîné
base_model.trainable = True

# On choisit à partir de quelle couche on commence à entraîner
fine_tune_at = 100  # tu peux ajuster
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompiler avec un learning rate plus petit
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fine-tuning sur quelques époques
fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stop]
)
model.save("mon_modele_fine_tune.h5")  # ou .keras (format moderne)

# === 7. Évaluation sur test set ===
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy après fine-tuning : {test_acc*100:.2f}%")
print(f"\n✅ Test loss après fine-tuning : {test_loss:.2f}")
# === 8. Affichage des courbes ===
plt.figure(figsize=(12, 5))

# Courbes de loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train (TL)', color='blue')
plt.plot(history.history['val_loss'], label='Val (TL)', color='orange')
plt.plot(fine_tune_history.history['loss'], label='Train (FT)', color='blue', linestyle='--')
plt.plot(fine_tune_history.history['val_loss'], label='Val (FT)', color='orange', linestyle='--')
plt.title('Loss pendant Transfert Learning + Fine-Tuning')
plt.xlabel('Époques')
plt.ylabel('Loss')
plt.legend()

# Courbes d’accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train (TL)', color='blue')
plt.plot(history.history['val_accuracy'], label='Val (TL)', color='orange')
plt.plot(fine_tune_history.history['accuracy'], label='Train (FT)', color='blue', linestyle='--')
plt.plot(fine_tune_history.history['val_accuracy'], label='Val (FT)', color='orange', linestyle='--')
plt.title('Accuracy pendant Transfert Learning + Fine-Tuning')
plt.xlabel('Époques')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
