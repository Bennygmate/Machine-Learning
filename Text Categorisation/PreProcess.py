# Benjamin Cheung
import sys
import os
import glob
import sklearn.datasets
from colorama import init
from termcolor import colored
from random import sample
import random
import numpy as np

def main():
    init()
    # get the dataset
    path = "20news-18828"
    # Create new dataset for processing
    cmd = "cp -r " + path + " dataset_process"
    print colored("Copying dataset into dataset_process", 'blue', attrs=['bold'])
    os.system(cmd)
    new_path = "dataset_process"
    change_incompatible_files(new_path)
    monte_carlo_cross(new_path)
    os.system("rm -r dataset_process")
    

def monte_carlo_cross(path):
    # Monte Carlo Cross Validation
    os.system("mkdir cross_valid")
    # Copy path into train and test dataset
    os.makedirs(os.path.join("cross_valid", "1"))
    os.makedirs(os.path.join("cross_valid/1", "train"))
    os.makedirs(os.path.join("cross_valid/1", "test"))
    os.makedirs(os.path.join("cross_valid", "2"))
    os.makedirs(os.path.join("cross_valid/2", "train"))
    os.makedirs(os.path.join("cross_valid/2", "test"))
    os.makedirs(os.path.join("cross_valid", "3"))
    os.makedirs(os.path.join("cross_valid/3", "train"))
    os.makedirs(os.path.join("cross_valid/3", "test"))
    os.makedirs(os.path.join("cross_valid", "4"))
    os.makedirs(os.path.join("cross_valid/4", "train"))
    os.makedirs(os.path.join("cross_valid/4", "test"))
    os.makedirs(os.path.join("cross_valid", "5"))
    os.makedirs(os.path.join("cross_valid/5", "train"))
    os.makedirs(os.path.join("cross_valid/5", "test"))
    os.makedirs(os.path.join("cross_valid", "6"))
    os.makedirs(os.path.join("cross_valid/6", "train"))
    os.makedirs(os.path.join("cross_valid/6", "test"))
    os.makedirs(os.path.join("cross_valid", "7"))
    os.makedirs(os.path.join("cross_valid/7", "train"))
    os.makedirs(os.path.join("cross_valid/7", "test"))
    os.makedirs(os.path.join("cross_valid", "8"))
    os.makedirs(os.path.join("cross_valid/8", "train"))
    os.makedirs(os.path.join("cross_valid/8", "test"))
    os.makedirs(os.path.join("cross_valid", "9"))
    os.makedirs(os.path.join("cross_valid/9", "train"))
    os.makedirs(os.path.join("cross_valid/9", "test"))
    os.makedirs(os.path.join("cross_valid", "10"))
    os.makedirs(os.path.join("cross_valid/10", "train"))
    os.makedirs(os.path.join("cross_valid/10", "test"))
    for f in range(1,11):
        pathing = "cross_valid/%d" %(f)
        V = os.listdir(path)
        for vj in V:
            tmp_path = path + "/" + vj
            tmp_pathing = pathing + "/train/" + vj
            tmp_pathings = pathing + "/test/" + vj
            cmd_1 = "mkdir " + tmp_pathing
            cmd_2 = "mkdir " + tmp_pathings
            os.system(cmd_1)
            os.system(cmd_2)
            folders = glob.glob(os.path.join(tmp_path, '*'))
            train_split = int(round(len(folders) * 0.6))
            indices = sample(range(0, len(folders)-1), train_split)
            i = 0
            folder_indices = len(folders) -1
            while i <= folder_indices:
                if i in indices:
                    cmd_train = "cp " + folders[i] + " " + tmp_pathing + "/" + str(i)
                    os.system(cmd_train)
                else:
                    cmd_test = "cp " + folders[i] + " " + tmp_pathings + "/" + str(i)
                    os.system(cmd_test)
                i += 1
            print colored("Made train and test for:", 'blue', attrs=['bold'])
            print vj
        print colored("Made train and test for cross-valid dataset:", 'blue', attrs=['bold'])
        print f


def change_incompatible_files(path):
    # find incompatible files
    print colored('Finding files incompatible with utf8: ', 'green', attrs=['bold'])
    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    files = sklearn.datasets.load_files(path)
    incompatible_files = []
    for i in range(len(files.filenames)):
        try:
            count_vector.fit_transform(files.data[i:i + 1])
        except UnicodeDecodeError:
            incompatible_files.append(files.filenames[i])
        except ValueError:
            pass
    print colored(len(incompatible_files), 'yellow'), 'files found'
    # delete them
    if(len(incompatible_files) > 0):
        print colored('Converting incompatible files', 'red', attrs=['bold'])
        for f in incompatible_files:
            print colored("Changing file to UTF-8:", 'red'), f
            cmd = "iconv -f ISO-8859-1 " + f + " -t UTF-8 -o tmp"
            cmdd = "cp tmp " + f
            os.system(cmd)
            os.system(cmdd)
            os.remove("tmp")

main()
