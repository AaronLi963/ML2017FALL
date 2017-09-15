import sys

f = open(sys.argv[1] , 'r')
data = f.read().strip('\n')
seperate = data.split(" ")
f.close()

f = open('Q1.txt' , 'w')
#print(seperate)
my_dict = {}
my_list = []
total = 0
most = 0
for i in seperate:
    if my_dict.get(i):
        my_dict[i] = my_dict.get(i) + 1
    else:
        my_dict[i] = 1
        my_list.append(i)
        total =total+1
    
    if my_dict[i] > most:
        most = my_dict[i]
#print(my_dict) 
counter = 0
for i in my_list:
    if counter == total-1:
        f.write('{0} {1} {2}'.format(i , counter , my_dict[i]))
    else:
        f.write('{0} {1} {2}\n'.format(i , counter , my_dict[i]))
    counter = counter+1

"""
counter = 0
while most > 0:
    i = 0
    while i < total:
        if my_dict.get(my_list[i]) == most:
            #print('{0} {1} {2}'.format(my_list[i] , counter , most))
            if counter == total-1:
                f.write('{0} {1} {2}'.format(my_list[i] , counter , most))
            else:
                f.write('{0} {1} {2}\n'.format(my_list[i] , counter , most))
            counter = counter + 1
        i = i + 1
    most = most - 1
"""
f.close()

#f.write(my_dict)
#f.close()