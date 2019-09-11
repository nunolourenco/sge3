import random
for i in range(30):
    data = ""
    for j in range(9):
        data += "%d\t" % random.randint(21,31)
    data += "%d" % random.randint(21,31)
    print data
