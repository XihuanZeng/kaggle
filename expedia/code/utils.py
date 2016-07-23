# utility functions

def get_attributes(file, attribute):
    f = open(file)
    header = f.readline()
    header = header.split(',')
    idx = header.index(attribute)
    arr = []
    while 1:
        line = f.readline()
        if line == '':
            break
        arr.append(line[idx])
    f.close()
    return arr



file = '/home/xihuan/gitrepos/kaggle/expedia/data/train.csv'
attribute = 'user_id'

u = get_attributes(file, attribute)