#!/usr/bin/env bash

cd "$(dirname "$0")"

echo "Cloning NewsQA repo to newsqa..."
git clone https://github.com/Maluuba/newsqa.git

cd newsqa
pip2 install --requirement requirements.txt

# download cnn
echo "Download the CNN stories manually to the maluuba/newsqa folder from (don't extract): http://cs.nyu.edu/~kcho/DMQA/"
echo "Press [Enter] when done so.."
read a

echo "Download the questions and answers to the maluuba/newsqa folder manually from: https://datasets.maluuba.com/NewsQA/dl..."
echo "Press [Enter] when done so.."
read a

cd maluuba/newsqa
tar xzf newsqa-data-v1.tar.gz

cd ../..

# fix a bug
sed -ie 's/\\r/\\n/g' maluuba/newsqa/data_processing.py
rm maluuba/newsqa/data_processing.pye

python2 maluuba/newsqa/example.py
python2 maluuba/newsqa/split_dataset.py

mv newsqa/maluuba/newsqa/* .
rm -r newsqa

echo "Find resulting dataset in data/NewsQA/newsqa/maluuba/newsqa/[train,dev,test]_story_ids.csv and newsqa/maluuba/newsqa/split_data"
echo "These can be used as input to the conversion scripts in jack/io/NewsQA2*.py"
