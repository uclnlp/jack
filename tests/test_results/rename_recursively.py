import sys
import os

path = sys.argv[1]
execute = False
try:
    execute = sys.argv[2] == '1'
except:
    pass

root_dir = os.getcwd()

files = []
dirs = []
for root, directories, filenames in os.walk(path):
    for filename in filenames:
            dirs.append(os.path.join(root_dir, root))
            files.append(os.path.join(root_dir, root, filename))


for f in files:
    if 'expected_results.txt' in f:
        if execute:
            os.remove(f)

for f, d in zip(files, dirs):
    if not 'expected_results.txt' in f:
        if execute:
            os.rename(f,os.path.join(d,'expected_results.txt'))
        else:
            print('{0} --> {1}'.format(f,
                os.path.join(d,'expected_results.txt')))
