# Natural Language Inference

The task is to predict whether a *hypothesis* is entailed by, contradicted by, or neutral with respect to a given *premise*.
In Jack, NLI is viewed as an instance of multiple choice Question Answering problem, by casting the hypothesis as the question, and the premise as the support.
The answer candidates to this question are the three possible outcomes or classes -- namely *entails*, *contradicts* or *neutral*.

### Pre-trained Models


#### SNLI

| Model         |  Accuracy  | Download        |
|---------------|------------|-----------------|
| DAM           |  84.6      |  [dam]   |
| ESIM          |  87.2      |  [esim]   |

[dam]: http://data.neuralnoise.com/jack/natural_language_inference/dam.zip
[esim]: http://data.neuralnoise.com/jack/natural_language_inference/esim.zip
