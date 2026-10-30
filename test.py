import re

answer = "Bảo hành 18 tháng [chunk_2], đổi trong 14 ngày [chunk_3]."
numbers = re.findall(r"\[chunk_(\d+)\]", answer)

print(numbers)