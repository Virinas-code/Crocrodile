"""
NN Random.

Reset NN wa, wb and wc.
"""
import csv
import random

def array_to_csv(array, csv_path):
    """Python array to CSV file."""
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in array:
            writer.writerow(row)
        file.close()
    return 0

continuer = input("Do you REALLY want to reset wa, wb and wc ? [y/N] ")

if continuer == "y":
    wa = []
    for a in range(74):
        wa.append([])
        for b in range(38):
            wa[a].append((random.random() * 2 - 1))
    wc = []
    for a in range(38):
        wc.append([])
        for b in range(1):
            wc[a].append((random.random() * 2 - 1))
    wb = []
    for a in range(38):
        wb.append([])
        for b in range(38):
            wb[a].append((random.random() * 2 - 1))
    array_to_csv(wa, "wa.csv")
    array_to_csv(wb, "wb.csv")
    array_to_csv(wc, "wc.csv")
    print("Done.")
