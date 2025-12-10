import re 

date = "2024-08-15 14:34:02"

match = re.match(r"[0-9]+[-][0-9]+[-][0-9]+\s?[0-9]+[:]", date)
print(match.group(0))  # Output: 2024-08-15 14: