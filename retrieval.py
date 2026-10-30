import math

def retrieval(query, documents):
    query = query.replace("?", " ")
    documents = documents.replace("?", " ")
    print(query.lower().split())
    unique_documents = normalize(documents)
    unique_query = normalize(query)
    common_words = unique_documents & unique_query
    score = len(common_words)
    return score

def normalize(text):
    text =  text.lower()
    for mark in "?.,!":
        text = text.replace(mark, " ")
    
    text = text.split()
    unique_text = set(text)
    return unique_text

def find_best_document(query, documents):
    best_score = 0
    best_document = None
    for document in documents:
        result = retrieval(query, document)
        if result > best_score:
            best_score = result
            best_document = document
    return best_document

query = "SẢN PHẨM BẢO HÀNH BAO LÂU?"
documents = [
    "đơn hàng được giao trong 2 ngày",
    "sản phẩm được bảo hành trong 18 tháng",
    "sản phẩm được đổi trong 7 ngày"
]

a = [1, 2]
b = [2, 1]
def dot_product(a, b):
    total = 0
    for x, y in zip(a,b):
        total += x * y
    return total

def vector_norm(v):
    total = 0
    for i in v:
        total += i**2
    return math.sqrt(total)

def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Hai vector phải có cùng số chiều.")
    
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Không thể tính cosine với vector không.")
    result = dot_product(a, b) / (norm_a * norm_b)
    return result



if __name__ == "__main__":
    print(cosine_similarity([1, 0], [1, 0]))
    # Đặt các lệnh thử nghiệm còn lại ở đây