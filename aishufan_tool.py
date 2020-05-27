"""
This kernel adds name prefix to all file names in a folder.

This helps to label a file after renaming.

Author: Shufan Yang, shufany@gmail.com
Date: 25/10/2019
"""

import os
from pathlib import Path
import shutil
from tqdm import tqdm
import click

launch_folder = Path(os.getcwd())
script_folder = Path(os.path.dirname(os.path.realpath(__file__)))


def prefix_rename(src, name_prefix, ext='jpg'):
    """
    Add prefix to all file names
    :param src: file folder
    :param name_prefix: The prefix to add
    :param ext: file extension to filter
    :return: None
    """
    files = list(src.glob(f'**/*.{ext}'))
    total = len(files)
    with tqdm(total=total) as pbar:
        for i in files:
            target = Path(f'{i.parent}/{name_prefix}_{i.name}')
            i.rename(target)
            pbar.set_description(str(target))
            pbar.update(1)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version='1.0.0')
def fxc_cli():
    pass

@fxc_cli.command()
@click.option('-s', '--src-folder', required=True, help='source image folder, relative to current folder')
@click.option('-p', '--name_prefix', default='LabelB', help='file name prefix')
def prefix(**kwargs):
    src = script_folder.joinpath(kwargs['src_folder'])
    name_prefix = kwargs['name_prefix']
    prefix_rename(src, name_prefix)


if __name__ == '__main__':
    # Execute commands
    fxc_cli()
