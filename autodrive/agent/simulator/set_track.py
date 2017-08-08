#!/usr/bin/python

import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--track')
parser.add_argument('-n', '--id', type=int,default=0)
#parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")

args = parser.parse_args()


id = args.id

import os
if id <10:
    id='0'+str(id)
else:id =str(id)
torcs_dir = os.path.expanduser('~/.torcs_0'+id)
if not os.path.exists(torcs_dir):
    raise IOError("can not found torcs dir, please install torcs into ${HOME}/torcs")

xmlfile = torcs_dir + '/config/raceman/practice.xml'
# print("write ",xmlfile)
#xmlfile = os.path.expanduser('/usr/local/share/games/torcs/config/raceman/practice.xml')
tree = ET.parse(xmlfile)

root = tree.getroot()
d = root[1][1][0].attrib
d['val'] = args.track
root[1][1][0].attrib = d
tree.write(xmlfile)