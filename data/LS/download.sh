#!/usr/bin/env bash
wget http://www.dianamccarthy.co.uk/files/task10data.tar.gz
wget http://nlp.cs.swarthmore.edu/semeval/tasks/task10/data/trial.tar.gz
wget http://nlp.cs.swarthmore.edu/semeval/tasks/task10/data/test.tar.gz
wget http://nlp.cs.swarthmore.edu/semeval/tasks/task10/data/key.tar.gz
tar -xzf task10data.tar.gz
tar -xzf trial.tar.gz
tar -xzf test.tar.gz
tar -xzf key.tar.gz
curl -O -L https://raw.githubusercontent.com/gaurav324/English-Lexicalized-Text-Substituion/master/TaskTestData/test/lexsub_test_cleaned.xml
curl -O -L https://raw.githubusercontent.com/gaurav324/English-Lexicalized-Text-Substituion/master/TaskTestData/trial/lexsub_trial_cleaned.xml
mv lexsub_test_cleaned.xml ./test/
mv lexsub_trial_cleaned.xml ./trial/