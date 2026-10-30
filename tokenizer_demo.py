from tokenizers import Tokenizer
from chunking import chunk_text_by_tokens

tokenizer = Tokenizer.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
tokenizer.no_truncation()
tokenizer.no_padding()

sample = "Sản phẩm được bảo hành trong 18 tháng."
encoded = tokenizer.encode(sample, add_special_tokens=False)


for token, offset in zip(encoded.tokens, encoded.offsets):
    start = offset[0]
    end = offset[1]
    print(token, offset, sample[start:end])




print(sample[encoded.offsets[0][0]:encoded.offsets[3][1]])



offsets = encoded.offsets
print(offsets)
chunk_size = 4
overlap = 2
step = chunk_size - overlap

for i in range(0, len(offsets), step):
    start = offsets[i][0]
    last_token = min(i + chunk_size, len(offsets)) - 1
    end = offsets[last_token][1]
    print(sample[start:end].strip())
    if i + chunk_size >= len(offsets):
        break


chunks = chunk_text_by_tokens(sample, tokenizer, 4, 2)
print(chunks)