from numpy import sqrt, sum, square, zeros


def get_similarity(y_1, y_2):
    dist = y_1 - y_2
    dist = sqrt(sum(square(dist))) + 1e-20

    similarity = 1.0 / dist

    return similarity


def get_similarity_matrix(embeddings_matrix):
    n_embeddings = len(embeddings_matrix)
    similarity_matrix = zeros((n_embeddings, n_embeddings))

    for i in range(n_embeddings):
        for j in range(n_embeddings):
            emb_1, emb_2 = embeddings_matrix[i], embeddings_matrix[j]
            similarity_matrix[i, j] = get_similarity(emb_1, emb_2)

    return similarity_matrix


def get_similarity_for_value(value_index, neighbours_indexes, node_processor):
    similarities = [node_processor.similarity_matrix[value_index, neighbours_index]
                    for neighbours_index in neighbours_indexes]

    return sum(similarities)
