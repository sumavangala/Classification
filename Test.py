__author__ = 'suma'

import numpy

# aa_milne_arr = ['1', '2', '3', '4', '5']
# print numpy.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.2, 0.1])
#
# dictlist = [dict() for x in range(5)]
# dictlist[0]['suma'] = 'reddy'
# print dictlist[0]
#
# print numpy.random.random(5)
weights = [dict() for x in range(5)]
records = []

for x in range(1, 6, 1):
    weights[0][x] = 1*1.0/5
    records.append(x)
weights[0][3] = 2
weights[0][2] = 1
print weights[0]
print weights[0].values()
#print numpy.random.choice(records, 5, p=weights[0].values())