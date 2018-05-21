src = 'IRISData.txt'
dest = 'Data.txt'

if __name__ == '__main__':
    with open(src, 'r', encoding='utf-8') as sf:
        with open(dest, 'w+', encoding='utf-8') as df:
            for line in sf.readlines():
                new_line = line.replace('\t', ',')
                df.write(new_line)
