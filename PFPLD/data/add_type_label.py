
ori_file = '/home/unaguo/hanson/data/landmark/WFLW191104/train_data/300W_LP.txt'
save_file = '/home/unaguo/hanson/data/landmark/WFLW191104/train_data/300W_LP1.txt'
lable = '0'
ori_lines = []
with open(ori_file, 'r')as f:
    ori_lines = f.readlines()

with open(save_file, 'w')as f:
    for line in ori_lines:
        line = line.strip()
        new_line = '{} {}\n'.format(line, lable)
        f.write(new_line)
