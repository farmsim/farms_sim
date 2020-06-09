"""View SPH"""

import glob
import sys
import math
import os
import os.path
import logging  # noqa: E402
from pyface.timer.api import do_later  # noqa: E402
from pysph.solver.utils import output_formats  # noqa: E402
from pysph.solver.utils import _sort_key  # noqa: E402
from pysph.tools.mayavi_viewer import (
    get_files_in_dir,
    sort_file_list,
    MayaviViewer,
    usage
)
if not os.environ.get('ETS_TOOLKIT'):
    # Set the default toolkit to qt4 unless the user has explicitly
    # set the default manually via the env var.
    from traits.etsconfig.api import ETSConfig
    ETSConfig.toolkit = 'qt4'

LOGGER = logging.getLogger()


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if '-h' in args or '--help' in args:
        usage()
        sys.exit(0)

    if '-v' in args:
        LOGGER.addHandler(logging.StreamHandler())
        LOGGER.setLevel(logging.INFO)
        args.remove('-v')

    kw = {}
    files = []
    scripts = []
    directory = None
    for arg in args:
        if '=' not in arg:
            if arg.endswith('.py'):
                scripts.append(arg)
                continue
            elif arg.endswith(output_formats):
                try:
                    _sort_key(arg)
                except ValueError:
                    print('Error: file name is not supported')
                    print('filename format accepted is *_number.npz'
                          ' or *_number.hdf5')
                    sys.exit(1)
                files.extend(glob.glob(arg))
                continue
            elif os.path.isdir(arg):
                directory = arg
                _files = get_files_in_dir(arg)
                files.extend(_files)
                config_file = os.path.join(arg, 'mayavi_config.py')
                if os.path.exists(config_file):
                    scripts.append(config_file)
                continue
            else:
                usage()
                sys.exit(1)
        key, arg = [x.strip() for x in arg.split('=')]
        try:
            val = eval(arg, math.__dict__)
            # this will fail if arg is a string.
        except Exception:
            val = arg
        kw[key] = val

    sort_file_list(files)
    live_mode = len(files) == 0

    # If we set the particle arrays before the scene is activated, the arrays
    # are not displayed on screen so we use do_later to set the  files.
    m = MayaviViewer(live_mode=live_mode)
    if not directory and len(files) > 0:
        directory = os.path.dirname(files[0])
    m.trait_set(directory=directory, trait_change_notify=False)
    do_later(m.trait_set, files=files, **kw)
    for script in scripts:
        do_later(m.run_script, script)
    print('Configure')
    m.configure_traits()


if __name__ == '__main__':
    main()
