import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import plot_confusion_matrix as cm
from sklearn.metrics import confusion_matrix

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

label_a = input('Escolha o primeiro algarismo do par que será treinado (0 a 9): ')
label_b = input('Escolha o segundo algarismo do par que será treinado (0 a 9): ')

# Filtrar apenas as imagens e rótulos com as duas labels escolhidas
train_filter = np.where((train_labels == int(label_a) ) | (train_labels == int(label_b)))
test_filter = np.where((test_labels == int(label_a)) | (test_labels == int(label_b)))

(train_images_filtered, train_labels_filtered) = train_images[train_filter], train_labels[train_filter]
(test_images_filtered, test_labels_filtered) = test_images[test_filter], test_labels[test_filter]

# Imprima as formas dos conjuntos de dados filtrados
print("Forma do conjunto de treinamento filtrado:", train_images_filtered.shape)
print("Forma do conjunto de rótulos de treinamento filtrado:", train_labels_filtered.shape)
print("Forma do conjunto de teste filtrado:", test_images_filtered.shape)
print("Forma do conjunto de rótulos de teste filtrado:", test_labels_filtered.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(28, (1, 1), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images_filtered, train_labels_filtered, epochs=10, 
                    validation_data=(test_images_filtered, test_labels_filtered))

plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.show()

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.show()

# Testing

classNames = [label_a, label_b]

x,y = (test_images_filtered[50], test_labels_filtered[50])
plt.imshow(x)
plt.axis('off')
plt.show()

# Expand the validation image to (1, 224, 224, 3) before predicting the label
prediction_scores = model.predict(np.expand_dims(x, axis=0))
predicted_index = np.argmax(prediction_scores)
print("True label: " + classNames[classNames.index(str(y))])
print("Predicted label: " + classNames[classNames.index(str(predicted_index))])

test_loss, test_acc = model.evaluate(test_images_filtered,  test_labels_filtered, verbose=2)
print(test_acc)


# Fazendo predição com os dados de teste
y_prediction = model.predict(test_images_filtered)
y_prediction = np.argmax (y_prediction, axis = 1)
y_test=test_labels_filtered

# Criando a matriz de confusão
result = confusion_matrix(y_test, y_prediction , normalize='pred')

cm.plotConfusionMatrix(result, classNames)

