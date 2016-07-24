"""
path = '/home/xihuan/Downloads/expedia/data/split_data/train/f0.csv'
f = open(path, 'rb')
header = f.readline().strip().split(',')
user_idx = header.index('user_id')
date_idx = header.index('date_time')
cnt = 0
users = []
while 1:
    line = f.readline()
    cnt += 1
    if cnt % 10000 == 0:
        print 'read %s lines' % cnt
    if line == '':
        break
    arr = line.split(',')
    if arr[date_idx][:4] != '2013':
        continue
    users.append(arr[user_idx])

users = list(set(users))
users.sort()
"""