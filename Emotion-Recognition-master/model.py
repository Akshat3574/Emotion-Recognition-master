import tensorflow as tf
import numpy as np


class ERModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        print("Initializing Emotion Recognition Model...")

        # First, try to explicitly register Sequential class
        try:
            # Try to load model directly (old approach)
            print("Attempting direct model loading...")
            with open(model_json_file, "r") as json_file:
                loaded_model_json = json_file.read()
                self.loaded_model = tf.keras.models.model_from_json(loaded_model_json)
                self.loaded_model.load_weights(model_weights_file)
                print("Model loaded successfully via json!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model with matching architecture...")
            self.loaded_model = self._create_model()

            try:
                # Try loading weights
                self.loaded_model.load_weights(model_weights_file)
                print("Weights loaded successfully!")
            except Exception as weight_err:
                print(f"Error loading weights: {weight_err}")
                print("Using model with random weights - accuracy will be poor")

        # Ensure model is compiled
        self.loaded_model.compile(optimizer='adam',
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])

        # Print model summary to verify architecture
        self.loaded_model.summary()

        # Test prediction to ensure model works
        test_input = np.random.rand(1, 48, 48, 1).astype(np.float32)
        test_output = self.loaded_model.predict(test_input, verbose=0)
        print(f"Test prediction shape: {test_output.shape}")
        print(f"Test prediction values: {test_output}")

    def _create_model(self):
        """Recreate the model architecture from the error message"""
        print("Creating new model architecture...")
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(128, (5, 5), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(512, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Dense(7, activation='softmax')
        ])
        return model

    def predict_emotion(self, img):
        """Predict emotion from image"""
        # Ensure image is properly preprocessed
        if img.max() > 1.0:
            img = img / 255.0

        self.preds = self.loaded_model.predict(img, verbose=0)
        print(f"Emotion predictions: {self.preds}")
        return ERModel.EMOTIONS_LIST[np.argmax(self.preds)]