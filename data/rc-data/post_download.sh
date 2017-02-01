#!/usr/bin/env bash
# download the data per instructions in README.md and then execute this script

tar -xvzf cnn_stories.tgz
tar -xvzf cnn.tgz
wget https://github.com/deepmind/rc-data/raw/master/generate_questions.py

# obtained from: https://github.com/deepmind/rc-data/blob/master/README.md
virtualenv venv
source venv/bin/activate
wget https://github.com/deepmind/rc-data/raw/master/requirements.txt
pip install -r requirements.txt
python generate_questions.py --corpus=cnn --mode=generate
python generate_questions.py --corpus=dailymail --mode=generate
deactivate