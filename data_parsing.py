import os,sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import project_statics

current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_dir, 'Dataset', 'train.csv')
test_path = os.path.join(current_dir, 'Dataset', 'test.csv')

from utils import parse_ourData_newformat

# raw file path, save destination path
parse_ourData_newformat(train_path, test_path, project_statics.Leading_indicator_pickle_files)