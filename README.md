# quebap
UCLMR KB and QA system/tools.

[Link](https://docs.google.com/document/d/1AaynDviR26bqofoImEcPxQgBcpvRBIcgLjScn-Hr6xk/edit) to Google Doc for overview/ inventory, etc.

#Completing the loop with MCTest
Let's say you have MCTest downloaded, and at least have two data files, mc160.train.tsv and mc160.train.ans, in your working directory.  We can convert these (together) into quebap format, via stdout, as follows:
```
python3 mctest2quebap.py mc160.train.tsv mc160.train.ans > mc160.quebap.json
```

With the data in the proper format, we can run a baseline model.  Here we use the random baseline (this assumes cwd is quebap/baseline, and we have copied over the above quebap.json file):
```
python3 random_baseline.py mc160.quebap.json > mc160.answers.json
```
The resulting json is just a list of answers, or more precisely, of candidates chosen as answers from the quebap.json file (so "label" fields will still be in these json entries).

How well did we do?  We can score this using the default evaluation script in quebap/eval:
```
python3 eval.py mc160.answers.json mc160.quebap.json
```
This will output a measure of rank accuracy.  The default is to score only the single best answer for each question (k=1), reproducing the standard accuracy used in classification and labeling tasks.
