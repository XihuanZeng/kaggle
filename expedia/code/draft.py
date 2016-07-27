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


"""
def generate_libffm_kth_model(data, dict_list, categorical_features, k):
    f = open('../data/libffm_data/train.txt', 'wb')
    g = open('../data/libffm_data/validate.txt', 'wb')
    for index, row in data.iterrows():
        is_booking = row['is_booking']
        if is_booking == 0:
            continue
        cluster = row['hotel_cluster']
        writable = ''
        for j in range(len(dict_list)):
            writable += '%s:%s:1 ' % (j, dict_list[j][str(row[categorical_features[j]])])
        if cluster == k:
            if row['month'] <= 10:
                f.write('1' + ' ' + writable + '\n')
            else:
                g.write('1' + ' ' + writable + '\n')
        else:
            if row['month'] <= 10:
                f.write('0' + ' ' + writable + '\n')
            else:
                g.write('0' + ' ' + writable + '\n')
    f.close()
    g.close()
"""