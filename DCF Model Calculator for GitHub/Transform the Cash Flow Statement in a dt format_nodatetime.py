import pandas as pd
import csv

filename = "RHHBY_Cash Flow.csv"

# Initialize an empty list to store the data
data = []
with open(filename, "r") as file:
    reader = csv.reader(file)
    headers = next(reader)

    for row in reader:
        converted_row = [row[0]]
        for i in range(1, len(row)):
            if row[i] in (".", "-"):
                converted_row.append(0.0)
            elif row[i] == "":
                converted_row.append(None)
            else:
                converted_row.append(float(row[i]))
        data.append(converted_row)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data, columns=headers)
#Eliminate the hyerarchical indentation
df['Year'] = df['Year'].str.strip()

print(df.head())

df.to_csv("RHHBY_Cash Flow_NEW_nodatetime.csv", index=False)