import json
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def preprocess_word(word):
    """Split camelCase and handle special characters."""
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


def semantic_similarity(word1, word2):
    """Calculate semantic similarity using WordNet."""
    word1 = preprocess_word(word1)
    word2 = preprocess_word(word2)

    syn1 = wn.synsets(word1)
    syn2 = wn.synsets(word2)

    if not syn1 or not syn2:
        words1 = word1.split()
        words2 = word2.split()
        similarities = []
        for w1, w2 in product(words1, words2):
            syn1 = wn.synsets(w1)
            syn2 = wn.synsets(w2)
            if syn1 and syn2:
                max_sim = max((s1.path_similarity(s2) or 0)
                              for s1, s2 in product(syn1, syn2))
                similarities.append(max_sim)
        return np.mean(similarities) if similarities else 0

    max_sim = max((s1.path_similarity(s2) or 0)
                  for s1, s2 in product(syn1, syn2))
    return max_sim


def extract_features(ontology_data):
    """Extract features from classes and relations."""
    classes = ontology_data['Classes']
    relations = ontology_data['Relations']

    # Class features
    class_features = []
    for class_name, attributes in classes.items():
        feature = f"{class_name}: {', '.join(attributes.keys())}"
        class_features.append(feature)

    # Relation features
    relation_features = []
    for relation_name, relation in relations.items():
        if isinstance(relation, dict):
            feature = f"{relation_name}: {relation['source']} -> {relation['target']} ({relation['multiplicity']})"
            relation_features.append(feature)
        elif isinstance(relation, list):
            for rel in relation:
                feature = f"{relation_name}: {rel['source']} -> {rel['target']} ({rel['multiplicity']})"
                relation_features.append(feature)

    return class_features, relation_features


def calculate_similarity_matrices(data):
    """Calculate both structural and semantic similarity matrices."""
    # Extract features
    prince2_class_features, prince2_relation_features = extract_features(data['PRINCE2'])
    scrum_class_features, scrum_relation_features = extract_features(data['Scrum'])

    # Structural similarity using TF-IDF
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))

    # Class similarity
    class_tfidf = vectorizer.fit_transform([' '.join(prince2_class_features),
                                            ' '.join(scrum_class_features)])
    structural_class_sim = cosine_similarity(class_tfidf[0:1], class_tfidf[1:2])[0][0]

    # Relation similarity
    relation_tfidf = vectorizer.fit_transform([' '.join(prince2_relation_features),
                                               ' '.join(scrum_relation_features)])
    structural_relation_sim = cosine_similarity(relation_tfidf[0:1], relation_tfidf[1:2])[0][0]

    # Semantic similarity matrix for classes
    prince2_classes = list(data['PRINCE2']['Classes'].keys())
    scrum_classes = list(data['Scrum']['Classes'].keys())

    semantic_matrix = pd.DataFrame(index=prince2_classes, columns=scrum_classes)
    for p_class in prince2_classes:
        for s_class in scrum_classes:
            sim = semantic_similarity(p_class, s_class)
            semantic_matrix.at[p_class, s_class] = round(sim, 3)

    return {
        'structural': {
            'class_similarity': structural_class_sim,
            'relation_similarity': structural_relation_sim,
            'overall_similarity': (structural_class_sim + structural_relation_sim) / 2
        },
        'semantic_matrix': semantic_matrix
    }


def analyze_ontology_statistics(data):
    """Calculate detailed statistics for both ontologies."""
    stats = {}
    for methodology in ['PRINCE2', 'Scrum']:
        ontology = data[methodology]
        stats[methodology] = {
            'num_classes': len(ontology['Classes']),
            'num_relations': sum(1 if isinstance(rel, dict) else len(rel)
                                 for rel in ontology['Relations'].values()),
            'avg_attributes_per_class': sum(len(attrs)
                                            for attrs in ontology['Classes'].values()) / len(ontology['Classes']),
            'attribute_types': defaultdict(int),
            'relation_multiplicities': defaultdict(int)
        }

        # Count attribute types
        for cls in ontology['Classes'].values():
            for attr_type in cls.values():
                stats[methodology]['attribute_types'][attr_type] += 1

        # Count relation multiplicities
        for rel in ontology['Relations'].values():
            if isinstance(rel, dict):
                stats[methodology]['relation_multiplicities'][rel['multiplicity']] += 1
            elif isinstance(rel, list):
                for r in rel:
                    stats[methodology]['relation_multiplicities'][r['multiplicity']] += 1

    return stats


def print_analysis_results(results, stats):
    """Print formatted analysis results."""
    print("\n=== Ontology Similarity Analysis ===")
    print("\nStructural Similarity Scores:")
    print(f"Overall Similarity: {results['structural']['overall_similarity']:.3f}")
    print(f"Class Structure Similarity: {results['structural']['class_similarity']:.3f}")
    print(f"Relation Structure Similarity: {results['structural']['relation_similarity']:.3f}")

    print("\nSemantic Similarity Matrix:")
    print(results['semantic_matrix'])

    print("\n=== Ontology Statistics ===")
    for methodology in ['PRINCE2', 'Scrum']:
        print(f"\n{methodology} Statistics:")
        methodology_stats = stats[methodology]
        print(f"Number of Classes: {methodology_stats['num_classes']}")
        print(f"Number of Relations: {methodology_stats['num_relations']}")
        print(f"Average Attributes per Class: {methodology_stats['avg_attributes_per_class']:.2f}")

        print("\nAttribute Types Distribution:")
        for attr_type, count in methodology_stats['attribute_types'].items():
            print(f"  {attr_type}: {count}")

        print("\nRelation Multiplicities Distribution:")
        for mult, count in methodology_stats['relation_multiplicities'].items():
            print(f"  {mult}: {count}")


# Main execution
if __name__ == "__main__":
    # Load JSON data
    with open('metamodel.json', 'r') as f:
        data = json.load(f)

    # Perform analysis
    similarity_results = calculate_similarity_matrices(data)
    statistics = analyze_ontology_statistics(data)

    # Print results
    print_analysis_results(similarity_results, statistics)

    # Save results
    similarity_results['semantic_matrix'].to_csv('semantic_similarity_matrix.csv')
    print("\nResults saved to 'semantic_similarity_matrix.csv'")