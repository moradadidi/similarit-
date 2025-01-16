import json
import numpy as np
import pandas as pd


# Load JSON Data
def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)


# Extract Feature Vector
def extract_feature_vector(data, key):
    """Extract feature vector for a given ontology."""
    classes = data[key]['Classes']
    relations = data[key]['Relations']

    class_count = len(classes)
    relation_count = len(relations)
    attributes_count = sum(len(attributes) for attributes in classes.values())

    return np.array([class_count, relation_count, attributes_count]), classes


# Normalize Feature Vectors
def normalize_vector(vec):
    """Normalize a feature vector to a 0-1 scale."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# Compute Euclidean Distance
def euclidean_distance(vec1, vec2):
    """Calculate the Euclidean distance between two vectors."""
    return np.linalg.norm(vec1 - vec2)


# Calculate Similarity Scores
def calculate_similarity(vector1, vector2):
    """Calculate structural similarity scores."""
    overall_similarity = 1 - euclidean_distance(vector1, vector2)
    return overall_similarity


# Generate Semantic Similarity Matrix
def generate_semantic_similarity_matrix(classes1, classes2):
    """Create a dummy semantic similarity matrix for demonstration."""
    # For simplicity, we're creating a random matrix here
    similarity_matrix = np.random.rand(len(classes1), len(classes2))
    return pd.DataFrame(similarity_matrix, index=classes1.keys(), columns=classes2.keys())


# Main Code
def main(file_path, key1, key2):
    # Load data
    data = load_json(file_path)

    # Extract feature vectors and class structures
    vector1, classes1 = extract_feature_vector(data, key1)
    vector2, classes2 = extract_feature_vector(data, key2)

    # Normalize vectors
    vector1_normalized = normalize_vector(vector1)
    vector2_normalized = normalize_vector(vector2)

    # Calculate Euclidean distance
    distance = euclidean_distance(vector1_normalized, vector2_normalized)

    # Calculate similarity score
    overall_similarity = calculate_similarity(vector1_normalized, vector2_normalized)

    # Generate Semantic Similarity Matrix
    semantic_similarity_matrix = generate_semantic_similarity_matrix(classes1, classes2)

    # Print results
    print("=== Ontology Similarity Analysis ===\n")
    print(f"Feature Vector for {key1}: {vector1}")
    print(f"Normalized Vector for {key1}: {vector1_normalized}")
    print(f"Feature Vector for {key2}: {vector2}")
    print(f"Normalized Vector for {key2}: {vector2_normalized}")
    print(f"The Euclidean Distance between {key1} and {key2} is: {distance:.4f}\n")
    print(f"Overall Similarity Score (1 - Euclidean Distance): {overall_similarity:.4f}\n")

    print("Semantic Similarity Matrix:\n")
    print(semantic_similarity_matrix)

    # Save the semantic similarity matrix to a CSV file
    semantic_similarity_matrix.to_csv('semantic_similarity_matrix.csv')
    print("Results saved to 'semantic_similarity_matrix.csv'")


# Run the script
if __name__ == "__main__":
    prince2_scrum_path = "metamodel.json"
    main(prince2_scrum_path, "PRINCE2", "Scrum")