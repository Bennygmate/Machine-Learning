import sys
import os
import glob
import sklearn.datasets
from colorama import init
from termcolor import colored

def main():
    init()
    # get the dataset
    print colored("Where is the dataset?", 'cyan', attrs=['bold'])
    arg = sys.stdin.readline()
    # remove any newlines or spaces at the end of the input
    path = arg.strip('\n')
    if path.endswith(' '):
        path = path.rstrip(' ')
    # create new dataset for processing
    cmd = "cp -r " + path + " dataset_process"
    print colored("Copying dataset into dataset_process", 'blue', attrs=['bold'])
    os.system(cmd)
    new_path = "dataset_process"
    change_incompatible_files(new_path)
    reorganize_dataset(new_path)
    #refine_all_emails(new_path)


def refine_all_emails(path):
    # load data
    print colored('Loading files into memory', 'green', attrs=['bold'])
    files = sklearn.datasets.load_files(path)
    # refine all emails
    print colored('Refining all files', 'green', attrs=['bold'])
    for i, email in zip(range(len(files.data)), files.data):
        files.data[i] = refine_single_email(email)


def refine_single_email(email):
    parts = email.split('\n')
    newparts = []
    # finished is when we have reached a line with something like 'Lines:' at the begining of it
    finished = False
    for part in parts:
        if finished:
            newparts.append(part)
            continue
        if not (part.startswith('Path:') or part.startswith('Newsgroups:') or part.startswith('Xref:')) and not finished:
            newparts.append(part)
            print(part)
            #print colored("NOT REFINED", 'blue', attrs=['bold'])
        if part.startswith('Lines:'):
            finished = True
    return '\n'.join(newparts)

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

def reorganize_dataset(path):
    likes = ['rec.sport.hockey', 'sci.crypt', 'sci.electronics']
    dislikes = ['sci.space', 'rec.motorcycles', 'misc.forsale']

    folders = glob.glob(os.path.join(path, '*'))
    if len(folders) == 2:
        return
    else:
        # create `likes` and `dislikes` directories
        if not os.path.exists(os.path.join(path, 'likes')):
            os.makedirs(os.path.join(path, 'likes'))
        if not os.path.exists(os.path.join(path, 'dislikes')):
            os.makedirs(os.path.join(path, 'dislikes'))

        for like in likes:
            files = glob.glob(os.path.join(path, like, '*'))
            for f in files:
                parts = f.split(os.sep)
                name = parts[len(parts) - 1]
                newname = like + '_' + name
                os.rename(f, os.path.join(path, 'likes', newname))
            os.rmdir(os.path.join(path, like))

        for like in dislikes:
            files = glob.glob(os.path.join(path, like, '*'))
            for f in files:
                parts = f.split(os.sep)
                name = parts[len(parts) - 1]
                newname = like + '_' + name
                os.rename(f, os.path.join(path, 'dislikes', newname))
            os.rmdir(os.path.join(path, like))

main()