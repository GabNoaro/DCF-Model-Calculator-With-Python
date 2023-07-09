import pandas as pd
import csv

filename = "RHHBY_Income Statement.csv"

# Initialize an empty list to store the data
data = []
with open(filename, "r") as file:
    reader = csv.reader(file)
    headers = next(reader)

    for row in reader:
        converted_row = [str(row[0])]
        for i in range(1, len(row)):
            if row[i] == "":
                converted_row.append(0.0) # 0.0 could be swapped with 0 because int occupies less memory than float
            else:
                converted_row.append(float(row[i]))
        data.append(converted_row)

# Fill all NaN values with 0
for row in data:
    for i in range(1, len(row)):
        if row[i] == 0.0:
            row[i] = None

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data[1:], columns=data[0])

# Convert the first column to datetime format
df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
