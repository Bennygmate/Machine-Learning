#!/usr/bin/env python
# Author: Bennygmate

import os

def exec_verbose(cmd):
	print(cmd)
	retval = os.system(cmd)
	return retval

from shutil import copyfile
if __name__ == "__main__":
    if os.name == 'nt': # Windows
        exec_verbose("copy ..\\Debug\\facetrain.exe .")
        exec_verbose("copy ..\\Debug\\hidtopgm.exe .")              

    exec_verbose("facetrain -n shades.net -t straightrnd_train.list -1 straightrnd_test1.list -2 straightrnd_test2.list -e 75")
    exec_verbose("facetrain -n face.net -t straightrnd_train.list -1 straightrnd_test1.list -2 straightrnd_test2.list -e 75")
    exec_verbose("facetrain -n face.net -t straighteven_train.list -1 straighteven_test1.list -2 straighteven_test2.list -e 100")
    exec_verbose("facetrain -n face.net -T -1 straighteven_test1.list -2 straighteven_test2.list")
    exec_verbose("facetrain -n pose.net -t all_train.list -1 all_test1.list -2 all_test2.list -e 100")
    exec_verbose("hidtopgm pose.net image-filename 32 30 n")
    print("\nThese are just skeleton code - implementations will follow this.\n")
    raw_input("Press any key to continue ...")