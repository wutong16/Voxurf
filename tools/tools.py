import os
from shutil import copyfile

def find_and_rename(root):
    dirs = os.listdir(root)
    if not os.path.exists(os.path.join(root, 'explore')):
        os.mkdir(os.path.join(root, 'explore'))
    for dir in dirs:
        if not os.path.isdir(os.path.join(root, dir)):
            continue
        name = os.path.join(root, dir, 'blended_images/00000000.jpg')
        copyfile(name, os.path.join(root, 'explore', dir + '_00000000.jpg'))

for n in l:
    if not os.path.exists(os.path.join('dataset_low_res', n[:-2])):
        print(n)
