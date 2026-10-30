from retrieval import cosine_similarity

def get_score(item):
    return item[0]

def retrieve(query, model, chunk_records, chunks_vectors, k = 2, min_score = 0.3):
    query_vector = list(model.embed(query))[0]
    result = []
    for record, vector in zip(chunk_records, chunks_vectors):
        cosine = cosine_similarity(query_vector, vector)
        if cosine >= min_score:
            result.append((cosine, record))

    sorted_result = sorted(result, key= get_score, reverse=True)
    return sorted_result[:k]