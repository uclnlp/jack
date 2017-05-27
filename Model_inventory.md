# Model Inventory

Below is an inventory of existing models in jtr (last updated: 27 May 2017). The current list of models can be obtained by running `python3 jtr/jack/train/train_reader.py --help`.
Models are defined in [jtr/readers.py](jtr/readers.py).

## snli_reader

The SNLI reader is defined as:

```
input_module = SingleSupportFixedClassInputs(shared_resources)
model_module = PairOfBiLSTMOverSupportAndQuestionModel(shared_resources)
output_module = EmptyOutputModule()
```

The reader is a multiple-choice model that encodes both the question and the support with a BiLSTM, concatenates the outputs of the BiLSTMs and
projects them into the number of classes (typically 3 in an SNLI setting: entailment, contradiction, neutral).
The model was developed for the [SNLI dataset](jtr/data/SNLI).


## fastqa_reader

The FastQA reader is defined as:

```
input_module = FastQAInputModule(shared_resources)
model_module = fatqa_model_module(shared_resources)
output_module = XQAOutputModule(shared_resources)
```

The reader is an extractive QA model, further documented in [Weissenborn et al. 2017](https://arxiv.org/abs/1703.04816).
It was developed for the [SQuAD dataset](jtr/data/SQuAD).


## cbow_xqa_reader

The cbow xqa reader is defined as:

```
input_module = CBOWXqaInputModule(shared_resources)
model_module = cbow_xqa_model_module(shared_resources)
output_module = XQANoScoreOutputModule(shared_resources)
```

The reader is a bag of word embeddings extractive QA model, further documented in [Weissenborn et al. 2017](https://arxiv.org/abs/1703.04816).
It was developed for the [SQuAD dataset](jtr/data/SQuAD).


## example_reader

The example reader is defined as:

```
input_module = SimpleMCInputModule(shared_resources)
model_module = SimpleMCModelModule(shared_resources)
output_module = SimpleMCOutputModule()
```

The reader is a multiple-choice QA model for questions, multiple supports and global candidates, that represents them as their averaged bag of word embeddings. Question and support representations are concatenated and are scored against a global set of candidates by taking the dot product between the question-support and the candidate representations.


## modelf_reader

The model F reader is defined as:

```
input_module = ModelFInputModule(shared_resources)
model_module = ModelFModelModule(shared_resources)
output_module = ModelFOutputModule()
```

The reader is a multiple-choice model for knowledge base population, described in [Riedel et al. 2013](www.aclweb.org/anthology/N13-1008) that uses questions and global candidates and no supports. Each question and candidate is viewed as one word, for which a representation is learned. Questions are scored against a global set of candidates by taking the dot product between the question and candidate representations.
The reader is developed for the [NYT dataset](jtr/data/NYT).