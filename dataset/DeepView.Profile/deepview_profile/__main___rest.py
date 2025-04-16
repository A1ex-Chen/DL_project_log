import warnings
warnings.filterwarnings("ignore", message="'deepview_profile.__main__' found in sys.modules after import of package 'deepview_profile', but prior to execution of 'deepview_profile.__main__'; this may result in unpredictable behaviour")

import argparse
import sys

import deepview_profile
import deepview_profile.commands.interactive
import deepview_profile.commands.memory
import deepview_profile.commands.time
import deepview_profile.commands.analysis




if __name__ == '__main__':
    main()