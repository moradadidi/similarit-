import json
import pandas as pd
from zss import Node, simple_distance


def build_tree(data: dict) -> Node:
    """
    Creates a tree of nodes from a nested dictionary.

    :param data: Nested dictionary representing the tree structure.
    :return: Root node of the tree.
    """
    if isinstance(data, dict):
        root_key = list(data.keys())[0]
        root_node = Node(root_key)
        build_subtree(root_node, data[root_key])
        return root_node
    raise ValueError("Data must be a dictionary.")


def build_subtree(parent: Node, data):
    """
    Recursively builds subtrees from the data and attaches them to the parent node.

    :param parent: Parent node to which children will be attached.
    :param data: Dictionary or list representing the subtree.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            child_node = Node(key)
            parent.addkid(child_node)
            build_subtree(child_node, value)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for key, value in item.items():
                    child_node = Node(key)
                    parent.addkid(child_node)
                    build_subtree(child_node, value)


def extract_structure_elements(data: dict, prefix='') -> dict:
    """
    Extracts classes and their properties from the metamodel structure.

    :param data: Nested dictionary containing the metamodel.
    :param prefix: Prefix for nested keys.
    :return: Dictionary of elements and their structures.
    """
    elements = {}

    if isinstance(data, dict):
        for key, value in data.items():
            current_key = f"{prefix}.{key}" if prefix else key
            elements[current_key] = value
            if isinstance(value, (dict, list)):
                elements.update(extract_structure_elements(value, current_key))

    return elements


def calculate_similarity_matrix(prince2_data: dict, scrum_data: dict) -> pd.DataFrame:
    """
    Calculates similarity matrix between PRINCE2 and Scrum elements.

    :param prince2_data: PRINCE2 metamodel data
    :param scrum_data: Scrum metamodel data
    :return: DataFrame containing similarity scores
    """
    # Extract classes and their structures
    prince2_elements = extract_structure_elements(prince2_data['PRINCE2'])
    scrum_elements = extract_structure_elements(scrum_data['Scrum'])

    # Initialize similarity matrix
    similarities = []
    prince2_keys = list(prince2_elements.keys())
    scrum_keys = list(scrum_elements.keys())

    # Calculate similarities
    for p_key in prince2_keys:
        row = []
        p_tree = build_tree({p_key: prince2_elements[p_key]})

        for s_key in scrum_keys:
            s_tree = build_tree({s_key: scrum_elements[s_key]})
            distance = simple_distance(p_tree, s_tree)
            similarity = 1 / (1 + distance)  # Similarity score calculation
            row.append(similarity)

        similarities.append(row)

    return pd.DataFrame(similarities, index=prince2_keys, columns=scrum_keys)


def main():
    # Load metamodel data
    try:
        with open('metamodel.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: 'metamodel.json' file not found. Please ensure the file exists.")
        return
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Please check the file format.")
        return

    # Calculate similarities
    df_similarity = calculate_similarity_matrix(data, data)  # Using same data for both since we only have one file

    # Filter relevant classes
    prince2_classes = ['BusinessCase', 'ProjectBoard', 'ProjectPlan', 'StagePlan', 'WorkPackage', 'EndStageReport']
    scrum_classes = ['ProductBacklog', 'Sprint', 'ScrumTeam', 'SprintBacklog', 'Increment', 'DailyScrum']

    filtered_df = df_similarity.loc[
        [f"Classes.{cls}" for cls in prince2_classes if f"Classes.{cls}" in df_similarity.index],
        [f"Classes.{cls}" for cls in scrum_classes if f"Classes.{cls}" in df_similarity.columns]
    ]

    # Display results
    print("\nFull Similarity Matrix:")
    print(df_similarity)
    print("\nFiltered Similarity Matrix (Main Classes):")
    print(filtered_df)

    # Calculate total similarity
    if not filtered_df.empty:
        total_similarity = filtered_df.values.mean()
        print(f"\nTotal Similarity between PRINCE2 and Scrum: {total_similarity:.4f}")
    else:
        print("\nNo matching classes found for comparison")


if __name__ == "__main__":
    main()