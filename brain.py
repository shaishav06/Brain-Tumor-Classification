import numpy as np
import pandas as pd
import os

base_path = "/kaggle/input/brain-tumor-multimodal-image-ct-and-mri/Dataset/Brain Tumor MRI images"
categories = ["Healthy","Tumor"]

image_paths = []
labels = []

for category in categories:
    category_path = os.path.join(base_path, category)
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        image_paths.append(image_path)
        labels.append(category)

df = pd.DataFrame({
    "image_path": image_paths,
    "label": labels
})

df.head()
df.tail()
df.shape
df.columns
df.duplicated().sum()
df.isnull().sum()
df.info()
df['label'].unique()
df['label'].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x="label", palette="viridis", ax=ax)

ax.set_title("Distribution of Tumor Types", fontsize=14, fontweight='bold')
ax.set_xlabel("Tumor Type", fontsize=12)
ax.set_ylabel("Count", fontsize=12)

plt.show()

import cv2

num_images = 5
plt.figure(figsize=(15, 12))

for i, category in enumerate(categories):
    category_images = df[df['label'] == category]['image_path'].iloc[:num_images]
    for j, img_path in enumerate(category_images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(len(categories), num_images, i * num_images + j + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(category)

plt.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['label'])
df = df[['image_path', 'category_encoded']]

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(df[['image_path']], df['category_encoded'])

df_resampled = pd.DataFrame(X_resampled, columns=['image_path'])
df_resampled['category_encoded'] = y_resampled

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_df_new, temp_df_new = train_test_split(df_resampled, train_size=0.8, shuffle=True, random_state=42, stratify=df_resampled['category_encoded'])
valid_df_new, test_df_new = train_test_split(temp_df_new, test_size=0.5, shuffle=True, random_state=42, stratify=temp_df_new['category_encoded'])

batch_size = 16
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

tr_gen = ImageDataGenerator(rescale=1./255)
ts_gen = ImageDataGenerator(rescale=1./255)

train_gen_new = tr_gen.flow_from_dataframe(train_df_new, x_col='image_path', y_col='category_encoded', target_size=img_size, class_mode='binary', batch_size=batch_size)
valid_gen_new = ts_gen.flow_from_dataframe(valid_df_new, x_col='image_path', y_col='category_encoded', target_size=img_size, class_mode='binary', batch_size=batch_size)
test_gen_new = ts_gen.flow_from_dataframe(test_df_new, x_col='image_path', y_col='category_encoded', target_size=img_size, class_mode='binary', batch_size=batch_size)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow import keras
from tensorflow.keras import layers

class AreaAttentionLayer(keras.layers.Layer):
    def __init__(self, reduction_ratio: int = 8, **kwargs):
        super(AreaAttentionLayer, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.conv1 = layers.Conv2D(32, kernel_size=1, activation='relu')
        self.conv2 = layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')
        self.bn = layers.BatchNormalization()

    def call(self, inputs):
        attention = self.conv1(inputs)
        attention = self.bn(attention)
        attention = self.conv2(attention)
        return inputs * attention

class CNNBlock(keras.layers.Layer):
    def __init__(self, filters: int, use_attention: bool = True, **kwargs):
        super(CNNBlock, self).__init__(**kwargs)
        self.filters = filters
        self.use_attention = use_attention
        if self.use_attention:
            self.area_attention = AreaAttentionLayer()
        self.conv = keras.Sequential([
            layers.Conv2D(filters, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

    def call(self, inputs):
        x = self.area_attention(inputs) if self.use_attention else inputs
        x = self.conv(x)
        return x + inputs

class RESNet(keras.Model):
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3, initial_filters: int = 32, **kwargs):
        super(RESNet, self).__init__(**kwargs)
        self.stem = keras.Sequential([
            layers.Conv2D(initial_filters, kernel_size=7, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        self.blocks = [CNNBlock(initial_filters * (2**i)) for i in range(3)]
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(dropout_rate)
        self.fc = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.stem(inputs)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = self.dropout(x)
        return self.fc(x)

def build_model(input_shape=(224, 224, 3), num_classes=2, learning_rate=1e-3):
    model = RESNet(num_classes=num_classes)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    model.build(input_shape=(None, *input_shape))
    return model

model = build_model()
history = model.fit(train_gen_new, validation_data=valid_gen_new, epochs=10, batch_size=16, verbose=1)

import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(test_gen_new)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()
y_true = test_gen_new.classes

conf_matrix = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print('Classification Report:\n', classification_report(y_true, y_pred_classes))