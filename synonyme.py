import json
import nltk
from nltk.corpus import wordnet as wn
from itertools import product
import pandas as pd
import numpy as np

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def load_json(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON.")
        return None


def extract_ontology_elements(data, key):
    """Extract ontology elements and convert set to sorted list for DataFrame compatibility."""
    classes = data[key]['Classes']
    relations = data[key]['Relations']

    elements = []
    # Add class names
    for class_name in classes.keys():
        elements.append(class_name)

    # Add relation names
    for relation_name in relations.keys():
        elements.append(relation_name)

    # Return sorted list instead of set
    return sorted(list(set(elements)))


def preprocess_word(word):
    """Preprocess word by splitting camelCase and removing special characters."""
    # Split camelCase
    words = []
    current_word = word[0]
    for c in word[1:]:
        if c.isupper():
            words.append(current_word.lower())
            current_word = c
        else:
            current_word += c
    words.append(current_word.lower())
    return ' '.join(words)


def word_similarity(word1, word2):
    """Compute semantic similarity between two words using WordNet."""
    # Preprocess words
    word1 = preprocess_word(word1)
    word2 = preprocess_word(word2)

    # Get synsets
    syn1 = wn.synsets(word1)
    syn2 = wn.synsets(word2)

    if not syn1 or not syn2:
        # Handle compound words
        words1 = word1.split()
        words2 = word2.split()

        # Calculate average similarity between all word pairs
        similarities = []
        for w1, w2 in product(words1, words2):
            syn1 = wn.synsets(w1)
            syn2 = wn.synsets(w2)
            if syn1 and syn2:
                max_sim = max((s1.path_similarity(s2) or 0)
                              for s1, s2 in product(syn1, syn2))
                similarities.append(max_sim)

        return np.mean(similarities) if similarities else 0

    # Calculate maximum similarity between all synset pairs
    max_sim = max((s1.path_similarity(s2) or 0)
                  for s1, s2 in product(syn1, syn2))

    return max_sim


def compute_similarity_matrix(elements1, elements2):
    """Compute pairwise similarity matrix between two sets of elements."""
    # Create DataFrame with lists instead of sets
    matrix = pd.DataFrame(index=elements1, columns=elements2)

    # Compute similarities
    for elem1 in elements1:
        for elem2 in elements2:
            similarity = word_similarity(elem1, elem2)
            matrix.at[elem1, elem2] = round(similarity, 3) if similarity else 0

    return matrix


def analyze_ontology_similarity(data):
    """Perform complete ontology similarity analysis."""
    if data is None:
        print("Error: No data to analyze.")
        return None

    # Extract elements
    prince2_elements = extract_ontology_elements(data, "PRINCE2")
    scrum_elements = extract_ontology_elements(data, "Scrum")

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(prince2_elements, scrum_elements)

    # Calculate statistics
    avg_similarity = similarity_matrix.mean().mean()
    max_similarity = similarity_matrix.max().max()
    min_similarity = similarity_matrix.min().min()

    # Find most similar pairs
    most_similar_pairs = []
    threshold = 0.5  # Adjust this threshold as needed

    for idx, row in similarity_matrix.iterrows():
        for col in similarity_matrix.columns:
            sim = similarity_matrix.at[idx, col]
            if sim >= threshold:
                most_similar_pairs.append((idx, col, sim))

    most_similar_pairs.sort(key=lambda x: x[2], reverse=True)

    return {
        'similarity_matrix': similarity_matrix,
        'statistics': {
            'average_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'most_similar_pairs': most_similar_pairs[:5]  # Top 5 most similar pairs
        }
    }


# Main execution
if __name__ == "__main__":
    # Load the JSON file
    file_path = "metamodel.json"
    data = load_json(file_path)

    if data:
        # Analyze data
        results = analyze_ontology_similarity(data)

        if results:
            # Print results
            print("\n=== Ontology Similarity Analysis ===\n")

            print("Similarity Matrix:")
            print(results['similarity_matrix'])
            print("\nStatistics:")
            stats = results['statistics']
            print(f"Average Similarity: {stats['average_similarity']:.3f}")
            print(f"Maximum Similarity: {stats['max_similarity']:.3f}")
            print(f"Minimum Similarity: {stats['min_similarity']:.3f}")

            print("\nTop Similar Element Pairs:")
            for pair in stats['most_similar_pairs']:
                print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")

            # Save results
            results['similarity_matrix'].to_csv("ontology_similarity_matrix.csv")
            print("\nResults have been saved to 'ontology_similarity_matrix.csv'")