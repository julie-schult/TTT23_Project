import matplotlib.pyplot as plt

import csv

roi1 = []
roi2 = []
roi3 = []
col = []

with open(r"C:\Users\sondr\OneDrive\Dokumenter\a\TTT23\TTT23_Project\Rawdata_values.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            print(row)
            roi1.append(float(row[1]))
            roi2.append(float(row[2]))
            roi3.append(float(row[3]))
            if "EXO" in row[0]:
                col.append((1, 0, 0))
            else:
                col.append((0, 0, 1))
            line_count += 1
    print(f'Processed {line_count} lines.')

plt.scatter(roi1, roi2, c=col)
plt.show()
plt.scatter(roi2, roi3, c=col)
plt.show()
plt.scatter(roi1, roi3, c=col)
plt.show()