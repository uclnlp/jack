# quebap
UCLMR KB and QA system/tools.

[Link](https://docs.google.com/document/d/1AaynDviR26bqofoImEcPxQgBcpvRBIcgLjScn-Hr6xk/edit) to Google Doc for overview/ inventory, etc.

#Completing the loop with MCTest
Let's say you have MCTest downloaded, and at least have two data files, mc160.train.tsv and mc160.train.ans, in your working directory.  We can convert these (together) into quebap format as follows:
```
python3 mctest2quebap.py mc160.train.tsv mc160.train.ans > mc160.quebap.json
```
