# Link Prediction

A Knowledge Graph is a set of *(s, p, o)* triples, where *s, o* denote the *subject* and *object* of the triple, and *p*  denotes its *predicate*: each *(s, p, o)* triple denotes a fact, represented as a relationship of type *p* between entities *s* and *o*, such as (London, capital of, UK).

The *Link Prediction* task consists in identifying missing *(s, p, o)* triples that are likely to encode true facts.


#### WN18

| Model         |  MRR       |     Hits@3     |     Hits@10    |
|---------------|------------|----------------|----------------|
| DistMult      | 0.822      | 0.914          | 0.936          |
| Complex       | 0.941      | 0.936          | 0.947          |

#### WN18RR

| Model         |  MRR       |     Hits@3     |     Hits@10    |
|---------------|------------|----------------|----------------|
| DistMult      | 0.430      | 0.443          | 0.490          |
| Complex       | 0.440      | 0.461          | 0.510          |

#### FB15k-237

| Model         |  MRR       |     Hits@3     |     Hits@10    |
|---------------|------------|----------------|----------------|
| DistMult      | 0.241      | 0.263          | 0.419          |
| Complex       | 0.247      | 0.275          | 0.428          |

### Implementing new Models

A Neural Link Prediction model is fully defined by its scoring function.

In *Jack*, such scoring functions are defined in https://github.com/uclmr/jack/blob/master/jack/readers/link_prediction/scores.py

