import csv
import numpy as np

ifile = open('train.csv','r')
reader = csv.reader(ifile)
x = []
y = []
print()
print('Start..')
print()
i = 0
for row in reader:
    i+=1
    print('X : ' + row[0],end=' || ')
    print('Y : ' + row[1],end=' || ')
    print('i : ' + str(i))
print()
print('End..')