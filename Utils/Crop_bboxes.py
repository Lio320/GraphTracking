import os
import numpy as np

print(os.getcwd())

path = '../Pseudolabels/Video_ionut/FeaturesLabels/'

for folder in os.listdir(path):
    new_path = path + folder + '/labels/'
    for file in os.listdir(new_path):
        file = new_path + file
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                line = [float(i) for i in line]
                x_center = line[1]
                y_center = line[2]
                width = line[3]
                height = line[4]

                if x_center + width/2 > 1:
                    print('error', x_center + width / 2)
                    width = width - (x_center + (width / 2) - 1)
                    print(x_center + width / 2)
                if x_center - width/2 < 0:
                    print('error', x_center - width/2)
                    width = width - (x_center - (width / 2))
                    print(x_center - width/2)
                if y_center + height/2 > 1:
                    print('error', y_center + height / 2)
                    height = height - (y_center + (height / 2) - 1)
                    print(y_center + height / 2)
                if y_center - height/2 < 0:
                    print('error', y_center - height/2)
                    height = height - (y_center - (height / 2))
                    print(y_center - height/2)







