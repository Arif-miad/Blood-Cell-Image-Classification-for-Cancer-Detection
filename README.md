
# Blood Cell Image Classification for Cancer Detection

This repository contains a complete workflow for classifying blood cell images into distinct categories using deep learning techniques. The goal is to detect and classify abnormal blood cells to assist in leukemia diagnosis.

## Dataset Overview

The dataset consists of **5,000 high-resolution images** categorized into the following classes:
- **Basophil**
- **Erythroblast**
- **Monocyte**
- **Myeloblast**
- **Segmented Neutrophil (seg_neutrophil)**

Each image is captured under **Wright-Giemsa staining** with a resolution of at least 1024x1024 pixels, ensuring high-quality morphology details. Expert hematopathologists have validated the dataset to guarantee accuracy and reliability.

### Key Features:
- **Resolution:** 1024x1024 pixels
- **Color:** 24-bit RGB
- **Magnification:** 100x oil immersion (1000x total)
- **Annotations:** Nuclear contours, cytoplasmic features, and abnormal inclusions (e.g., Auer rods)

---

## Repository Structure
```
blood-cell-image-classification/
├── dataset/                   # Sample dataset (images by class)
├── notebooks/                 # Jupyter notebooks for EDA and model implementation
├── src/                       # Python scripts for data processing and modeling
├── models/                    # Pre-trained models
├── results/                   # Model performance metrics and predictions
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```
## Code Implementation

2. **Import Necessary Libraries**:
   ```python
   import os
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   import cv2
   from sklearn.model_selection import train_test_split
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   from tensorflow.keras.utils import to_categorical
   from sklearn.metrics import classification_report, confusion_matrix
   ```
3. **Define Dataset Path**:
   ```python
   DATASET_PATH = '/path/to/dataset/'
   ```
4. **Set Random Seed**:
   ```python
   SEED = 42
   np.random.seed(SEED)
   ```
5. **List Classes**:
   ```python
   CLASSES = ['basophil', 'erythroblast', 'monocyte', 'myeloblast', 'seg_neutrophil']
   ```
6. **Verify Dataset Structure**:
   ```python
   for class_name in CLASSES:
       print(f"{class_name}: {len(os.listdir(os.path.join(DATASET_PATH, class_name)))} images")
   ```
7. **Check for Missing or Corrupt Files**:
   ```python
   for class_name in CLASSES:
       for img_file in os.listdir(os.path.join(DATASET_PATH, class_name)):
           try:
               img = cv2.imread(os.path.join(DATASET_PATH, class_name, img_file))
           except:
               print(f"Corrupt file: {img_file}")
   ```
8. **Define Image Size**:
   ```python
   IMG_HEIGHT, IMG_WIDTH = 128, 128
   ```
9. **Initialize Empty Arrays for Data and Labels**:
   ```python
   images = []
   labels = []
   ```
10. **Load Images into Arrays**:
   ```python
   for class_index, class_name in enumerate(CLASSES):
       folder_path = os.path.join(DATASET_PATH, class_name)
       for img_file in os.listdir(folder_path):
           img = cv2.imread(os.path.join(folder_path, img_file))
           img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
           images.append(img)
           labels.append(class_index)
   images = np.array(images)
   labels = np.array(labels)
   ```

---

### **11-20: Data Preprocessing**
11. **Normalize Image Data**:
   ```python
   images = images / 255.0
   ```
12. **Convert Labels to Categorical Format**:
   ```python
   labels = to_categorical(labels, num_classes=len(CLASSES))
   ```
13. **Split Data into Train, Validation, and Test Sets**:
   ```python
   X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=SEED)
   X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)
   ```
14. **Initialize Image Data Generator for Augmentation**:
   ```python
   datagen = ImageDataGenerator(
       rotation_range=15,
       width_shift_range=0.1,
       height_shift_range=0.1,
       shear_range=0.1,
       zoom_range=0.1,
       horizontal_flip=True
   )
   datagen.fit(X_train)
   ```
15. **Display Augmented Images**:
   ```python
   augmented_images = next(datagen.flow(X_train, y_train, batch_size=4))
   fig, axes = plt.subplots(1, 4, figsize=(10, 5))
   for i, img in enumerate(augmented_images[0]):
       axes[i].imshow(img)
       axes[i].axis('off')
   plt.show()
   ```
![display_image](https://github.com/Arif-miad/Blood-Cell-Image-Classification-for-Cancer-Detection/blob/main/image/blo2.PNG)
---

### **21-30: Model Building and Training**
16. **Define the Model Architecture**:
   ```python
   model = Sequential([
       Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
       MaxPooling2D(pool_size=(2, 2)),
       BatchNormalization(),
       Conv2D(64, (3, 3), activation='relu'),
       MaxPooling2D(pool_size=(2, 2)),
       BatchNormalization(),
       Conv2D(128, (3, 3), activation='relu'),
       MaxPooling2D(pool_size=(2, 2)),
       BatchNormalization(),
       Flatten(),
       Dense(256, activation='relu'),
       Dropout(0.5),
       Dense(len(CLASSES), activation='softmax')
   ])
   ```
17. **Compile the Model**:
   ```python
   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   ```
18. **Display Model Summary**:
   ```python
   model.summary()
   ```
19. **Train the Model**:
   ```python
   history = model.fit(
       datagen.flow(X_train, y_train, batch_size=32),
       validation_data=(X_val, y_val),
       epochs=20,
       steps_per_epoch=len(X_train) // 32
   )
   ```
20. **Plot Training Performance**:
   ```python
   plt.plot(history.history['accuracy'], label='Train Accuracy')
   plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
   plt.legend()
   plt.show()
   ```
![train](https://github.com/Arif-miad/Blood-Cell-Image-Classification-for-Cancer-Detection/blob/main/image/blo3.PNG)
---

### **31-40: Evaluation and Deployment**
21. **Evaluate Model on Test Data**:
   ```python
   test_loss, test_accuracy = model.evaluate(X_test, y_test)
   print(f"Test Accuracy: {test_accuracy:.2f}")
   ```
22. **Generate Predictions**:
   ```python
   y_pred = model.predict(X_test)
   y_pred_classes = np.argmax(y_pred, axis=1)
   y_true = np.argmax(y_test, axis=1)
   ```
23. **Display Classification Report**:
   ```python
   print(classification_report(y_true, y_pred_classes, target_names=CLASSES))
   ```
![classification report](https://github.com/Arif-miad/Blood-Cell-Image-Classification-for-Cancer-Detection/blob/main/image/blod4.PNG)


24. **Plot Confusion Matrix**:
   ```python
   cm = confusion_matrix(y_true, y_pred_classes)
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
   plt.xlabel('Predicted')
   plt.ylabel('Actual')
   plt.show()
   ```
25. **Save the Trained Model**:
   ```python
   model.save('blood_cell_classifier.h5')
   ```
26. **Load the Saved Model**:
   ```python
   from tensorflow.keras.models import load_model
   loaded_model = load_model('blood_cell_classifier.h5')
   ```
27. **Test with a New Image**:
   ```python
   img = cv2.imread('/path/to/test/image.jpg')
   img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)) / 255.0
   img = np.expand_dims(img, axis=0)
   prediction = np.argmax(loaded_model.predict(img))
   print(f"Predicted Class: {CLASSES[prediction]}")
   ```


## Results

### **Model Performance**
- **Training Accuracy:** 97%
- **Validation Accuracy:93%
- **Test Accuracy:** 98%

### **Visualizations**
- **Sample Predictions:**
- 
  ![Prediction Example](https://github.com/Arif-miad/Blood-Cell-Image-Classification-for-Cancer-Detection/blob/main/image/blod.PNG)
  
- **Confusion Matrix:**
- 
  ![Confusion Matrix](https://github.com/Arif-miad/Blood-Cell-Image-Classification-for-Cancer-Detection/blob/main/image/blo1.PNG)

---


## Future Enhancements
- Add multi-class support for additional cell types.
- Experiment with advanced architectures like Vision Transformers.
- Develop a web application for real-time leukemia detection.

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- Thanks to the creators of the blood cell image dataset.
- Wright-Giemsa staining technique for high-quality image preparation.
- Open-source contributors for machine learning libraries.

