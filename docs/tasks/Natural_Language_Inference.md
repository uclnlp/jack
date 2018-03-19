# Natural Language Inference

The task is to predict whether a *hypothesis* is entailed by, contradicted by, or neutral with respect to a given *premise*.
In Jack, NLI is viewed as an instance of multiple choice Question Answering problem, by casting the hypothesis as the question, and the premise as the support.
The answer candidates to this question are the three possible outcomes or classes -- namely *entails*, *contradicts* or *neutral*.

### Pre-trained Models

 
#### WN18

| Model         |  MRR  | Hits@10  | Download        |
|---------------|-------|----------|-----------------|
| DistMult      |       |          |  [distmult_fb]  |
| ComplEx       |       |          |  [complex_fb]   |

[distmult]:
[complex]:

#### FB15K

| Model         |  MRR  | Hits@10  | Download        |
|---------------|-------|----------|-----------------|
| DistMult      |       |          |  [distmult_fb]  |
| ComplEx       |       |          |  [complex_fb]   |

[distmult_fb]:
[complex_fb]:


### Implementing new Models

TODO

### Supported Models

TODO

### Supported Datasets

TODO
