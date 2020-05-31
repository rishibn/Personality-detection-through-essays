import csv
r = csv.reader(open(r'D:\MCA matters\4th sem\minor\essays1.csv')) # Here your csv file
lines = list(r)
for line in lines:
    print(line)
    if line[2]=='y':
        line[2]=1
    else:
        line[2]=0
    if line[3]=='y':
        line[3]=1
    else:
        line[3]=0
    if line[4]=='y':
        line[4]=1
    else:
        line[4]=0
    if line[5]=='y':
        line[5]=1
    else:
        line[5]=0
    if line[6]=='y':
        line[6]=1
    else:
        line[6]=0
    print(line)

writer = csv.writer(open('essays2.csv', 'w'))
writer.writerows(lines)
