# GradioBrainTumor

````markdown
# 🧠 Brain Tumor Classification with CNN & Gradio (Google Colab)

This project performs **automatic classification of brain tumors** into four categories: `glioma`, `meningioma`, `pituitary`, and `notumor`, using a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras. It includes a **Gradio interface** for interactive image upload and prediction.

---

## 📌 Project Highlights

- 🧠 Classifies brain MRI images into 4 tumor types
- 🤖 Uses a CNN model trained from scratch
- 📊 Shows model accuracy and loss during training
- 🖼️ Image pre-processing and normalization
- 🧪 Test-train split for evaluation
- 🌐 Deployed with **Gradio** for easy image upload and prediction
- 💾 Dataset is loaded from **Google Drive**

---

## 🛠️ Installation

Install the required packages in Colab:

```python
!pip install gradio
!pip install tensorflow
!pip install opencv-python-headless
````

---

## 📂 Dataset

The dataset is expected to be structured as follows inside your Google Drive:

```
Brain Tumor Segmentation/
└── Training/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Update the path in the code if needed:

```python
DATASET_PATH = '/content/drive/MyDrive/Brain Tumor Segmentation/Training'
```

Mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🧠 Model Architecture

A sequential CNN model with:

* `Conv2D` layers for feature extraction
* `MaxPooling2D` for spatial reduction
* `BatchNormalization` for training stability
* `GlobalAveragePooling2D` to flatten features
* `Dense` layers for classification
* `Dropout` to prevent overfitting

Compiled with:

```python
loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy']
```

---

## 🔄 Data Preparation

* All images are resized to `128x128`
* Normalized to pixel range `[0, 1]`
* Labels are one-hot encoded using `LabelBinarizer`
* Split into train/test sets using `train_test_split`

---

## 📈 Training

Trains the model for `10 epochs` with a batch size of `16`.

You can change the following for faster training or better results:

```python
EPOCHS = 10
BATCH_SIZE = 16
```

---

## 🧪 Prediction Function

The function `predict_image(image)`:

* Preprocesses uploaded image
* Predicts tumor type using the trained model
* Returns the predicted label

---

## 🌐 Gradio Interface

Run this to launch the web-based interface:

```python
interface = gr.Interface(fn=predict_image,
                         inputs=gr.Image(type='pil'),
                         outputs=gr.Label(num_top_classes=4),
                         title="Brain Tumor Classifier",
                         description="Upload an image to classify a brain tumor as glioma, meningioma, notumor, or pituitary.")
interface.launch()
```

---

## 📊 Example Output

After training and launching the interface:

* Upload an MRI image
* The model returns the predicted tumor type
* Example: `Prediction: meningioma`

---

## ⚠️ Notes

* Ensure that your dataset is properly organized and loaded via Google Drive
* Training may take a few minutes depending on the dataset size
* Gradio is supported in Colab but may open in a new tab or give a public link

---

## 📃 License

This project is for educational and research use only. All libraries used are under their respective open-source licenses.

---

## 🙋‍♀️ Author

Developed using TensorFlow, OpenCV, Gradio, and Keras in a Colab environment.


