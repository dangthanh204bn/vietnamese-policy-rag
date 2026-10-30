def build_rerank_pairs(query, results):
    pairs = []
    for score, record in results:
        ls = [query, record["parent_text"]]
        pairs.append(ls)

    return pairs

def get_score(item):
    return item[0]

def rerank_results(query, results, reranker, top_n=3):
    if results == []:
        return []
    pairs = build_rerank_pairs(query, results)
    scores = reranker.predict(pairs)
    ls = []
    for old_result, new_score in zip(results, scores):
        record = old_result[1]
        ls.append((float(new_score),record))
    ranked_ls = sorted(ls, key=get_score, reverse=True)

    return ranked_ls[:top_n]
