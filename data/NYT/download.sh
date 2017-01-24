#!/usr/bin/env bash
wget -O data/NYT/naacl2013.txt.zip https://www.dropbox.com/s/5iulumlihydo1k7/naacl2013.txt.zip?dl=1
unzip data/NYT/naacl2013.txt.zip -d data/NYT/
rm data/NYT/naacl2013.txt.zip