# Link Prediction

A Knowledge Graph is a set of *(s, p, o)* triples, where *s, o* denote the *subject* and *object* of the triple, and *p* 
denotes its *predicate*: each *(s, p, o)* triple denotes a fact, represented as a relationship of type *p* between 
entities *s* and *o*, such as (London, capital of, UK).
The *Link Prediction* task consists in identifying missing *(s, p, o)* triples that are likely to encode true facts.

### Pre-trained Models

 
#### SNLI

| Model         |  Accuracy  |  Download       |
|---------------|------------|-----------------|
| DAM           |            |     [dam]       |
| ESIM          |            |     [esim]      |

[dam]:
[esim]:

#### MultiNLI

| Model         |  Matched   |  Mismatched  |  Download       |
|---------------|------------|--------------|-----------------|
| DAM           |            |              |     [dam_mnli]  |
| ESIM          |            |              |     [esim_mnli] |

[dam_mnli]:
[esim_mnli]:


### Implementing new Models

TODO

### Supported Models

TODO

### Supported Datasets

TODO
