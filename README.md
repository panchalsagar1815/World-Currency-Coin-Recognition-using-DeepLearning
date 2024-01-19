# World-Currency-Coin-Recognition-using-DeepLearning

#**Project Title: World Currency Coin Recognition using Deep Learning**

**Overview:**
The "World Currency Coin Recognition using Deep Learning" project aims to develop an advanced system for recognizing and classifying coins from different parts of the world based on their values and respective countries. Leveraging deep learning techniques, the project focuses on implementing a robust solution capable of accurately identifying both the value and origin of a given coin through image analysis.

**Key Features:**

1. **Data Collection and Cleaning:**
   - Compiled a diverse dataset containing images of coins from various countries, encompassing different values.
   - Employed data cleaning techniques to ensure dataset quality, removing any artifacts or inconsistencies in the coin images.
   - Ensured balanced representation of coins across different denominations and geographical origins.

2. **Input Pipeline and Preprocessing:**
   - Created an efficient input pipeline to handle the loading and preprocessing of coin images.
   - Applied preprocessing techniques to standardize and enhance the quality of the images, ensuring optimal model performance.

3. **Model Architecture - MobileNetV2:**
   - Utilized the MobileNetV2 architecture, a lightweight and efficient neural network model for image classification tasks.
   - Leveraged the pre-trained weights from the ImageNet dataset to capture general features from coin images.
   - Fine-tuned the model by allowing training on the specific coin recognition task.

4. **Classification Layer:**
   - Added a dense layer with 512 neurons and ReLU activation to extract high-level features from the MobileNetV2 base.
   - Applied dropout regularization to prevent overfitting and enhance the model's generalization capabilities.
   - The final layer with 211 neurons and softmax activation predicts the coin's class, representing both its value and the country it is used in.

**Model Architecture:**
```python
inp = Input(shape=(224, 224, 3))

# MobileNetV2 Model
model1 = MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_tensor=inp,
    input_shape=(224, 224, 3),
    pooling='avg')

for layer in model1.layers:
    layer.trainable = True  

# Classification layer
i = Dense(512, activation='relu')(model1.output)
i = Dropout(.8)(i)
pred = Dense(211, activation='softmax')(i)

# Final model
model = Model(inputs=model1.input, outputs=pred)
```

5. **Training Optimization - Early Stopping:**
   - Implemented early stopping during model training to prevent overfitting and optimize training time.
   - Monitored a selected metric (e.g., validation loss) and stopped training when no improvement was observed.

**Conclusion:**
The successful implementation of this deep learning model opens up possibilities for automating the recognition and classification of world currency coins, facilitating tasks such as coin counting and authentication. Future enhancements may involve expanding the dataset to include additional coin variations and exploring deployment in real-world scenarios.
