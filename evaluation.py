import re
def extract_citation_ids(answer):
    numbers = re.findall(r"\[chunk_(\d+)\]", answer)
    int_numbers = set()
    for i in numbers:
        int_numbers.add(int(i))
    return int_numbers

if __name__ == "__main__":
    answer = "Bảo hành 18 tháng [chunk_2], đổi trong 14 ngày [chunk_3]."
    print(extract_citation_ids(answer))