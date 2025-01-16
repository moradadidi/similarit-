import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import json


# Load JSON Data
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# Extract Feature Vector
def extract_feature_vector(data, key):
    classes = data[key]['Classes']
    relations = data[key]['Relations']

    class_count = len(classes)
    relation_count = len(relations)
    attributes_count = sum(len(attributes) for attributes in classes.values())

    return np.array([class_count, relation_count, attributes_count])


# Prepare Dataset for Siamese Network
def prepare_pairs(data):
    prince2_vector = extract_feature_vector(data, "PRINCE2")
    scrum_vector = extract_feature_vector(data, "Scrum")

    # Example: Positive (similar) and negative (dissimilar) pairs
    positive_pairs = [(prince2_vector, scrum_vector, 1)]
    negative_pairs = [
        (prince2_vector, prince2_vector + np.random.random(prince2_vector.shape), 0)
    ]

    pairs = positive_pairs + negative_pairs
    left = np.array([p[0] for p in pairs])
    right = np.array([p[1] for p in pairs])
    labels = np.array([p[2] for p in pairs])

    return left, right, labels


# Build Siamese Network
def build_siamese_network(input_shape):
    # Shared network
    input = Input(shape=input_shape)
    x = Dense(16, activation="relu")(input)
    x = Dense(8, activation="relu")(x)
    output = Dense(4, activation="relu")(x)
    shared_network = Model(input, output)

    # Siamese network
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = shared_network(input_a)
    processed_b = shared_network(input_b)

    # L1 Distance layer
    l1_distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
    similarity = Dense(1, activation="sigmoid")(l1_distance)

    model = Model([input_a, input_b], similarity)
    return model


# Main Code
prince2_scrum_path = "metamodel.json"
data = load_json(prince2_scrum_path)

# Prepare data
input_shape = (3,)  # Feature vector shape: [class_count, relation_count, attributes_count]
left, right, labels = prepare_pairs(data)

# Build and compile model
siamese_model = build_siamese_network(input_shape)
siamese_model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Train model
siamese_model.fit([left, right], labels, batch_size=2, epochs=10)

# Evaluate similarity
similarity_score = siamese_model.predict([np.expand_dims(left[0], axis=0), np.expand_dims(right[0], axis=0)])
print(f"Similarity score between PRINCE2 and Scrum: {similarity_score[0][0]:.2f}")
