# Model Inventory

Below is an inventory of existing models in jtr (last updated: 27 May 2017). The current list of models can be obtained by running `python3 jtr/jack/train/train_reader.py --help`.
Models are defined in `jtr/readers.py`. Once imported you can construct models by calling their respective factory methods given a SharedResources object (usually holds configuration and vocab, but can contain additional information). The following lists models and the individual modules they are composed of. Modules can be shared across readers. Note that after a model is constructed it is not yet setup. It can be setup by loading it from file (`reader.setup_from_file(...)` when working with trained models) or from datasets which is done in the training script.

## SNLI Readers

Jack the Reader has several SNLI readers:
- Conditional Bidirectional LSTM [Rocktäschel et al. 2016](https://arxiv.org/abs/1509.06664)
- Decomposable Attention Model (DAM) [Parikh et al. 2016](https://arxiv.org/abs/1606.01933)
- Enhanced LSTM (ESIM) [Chen et al. 2017](https://arxiv.org/abs/1609.06038)

The SNLI readers are defined as follows in `jtr/readers.py`:

```python
@__mcqa_reader
def cbilstm_snli_reader(shared_resources: SharedResources):
    """
    Creates a SNLI reader instance (multiple choice qa model).
    This particular reader uses a conditional Bidirectional LSTM, as described in [1].

    [1] Tim Rocktäschel et al. - Reasoning about Entailment with Neural Attention. ICLR 2016
    """
    from jtr.tasks.mcqa.simple_mcqa import MultiSupportFixedClassInputs, PairOfBiLSTMOverSupportAndQuestionModel,\
        EmptyOutputModule
    input_module = MultiSupportFixedClassInputs(shared_resources)
    model_module = PairOfBiLSTMOverSupportAndQuestionModel(shared_resources)
    output_module = EmptyOutputModule()
    return JTReader(shared_resources, input_module, model_module, output_module)


@__mcqa_reader
def dam_snli_reader(shared_resources: SharedResources):
    """
    Creates a SNLI reader instance (multiple choice qa model).
    This particular reader uses a Decomposable Attention Model, as described in [1].

    [1] Ankur P. Parikh et al. - A Decomposable Attention Model for Natural Language Inference. EMNLP 2016
    """
    from jtr.tasks.mcqa.simple_mcqa import MultiSupportFixedClassInputs, DecomposableAttentionModel, EmptyOutputModule
    input_module = MultiSupportFixedClassInputs(shared_resources)
    model_module = DecomposableAttentionModel(shared_resources)
    output_module = EmptyOutputModule()
    return JTReader(shared_resources, input_module, model_module, output_module)


@__mcqa_reader
def esim_snli_reader(shared_resources: SharedResources):
    """
    Creates a SNLI reader instance (multiple choice qa model).
    This particular reader uses an Enhanced LSTM Model (ESIM), as described in [1].

    [1] Qian Chen et al. - Enhanced LSTM for Natural Language Inference. ACL 2017
    """
    from jtr.tasks.mcqa.simple_mcqa import MultiSupportFixedClassInputs, ESIMModel, EmptyOutputModule
    input_module = MultiSupportFixedClassInputs(shared_resources)
    model_module = ESIMModel(shared_resources)
    output_module = EmptyOutputModule()
    return JTReader(shared_resources, input_module, model_module, output_module)

```

The reader is a multiple-choice model that encodes both the question and the support with a BiLSTM, concatenates the outputs of the BiLSTMs and
projects them into the number of classes (typically 3 in an SNLI setting: entailment, contradiction, neutral).
The model was developed for the SNLI dataset in `jtr/data/SNLI`.


## FastQA Reader

The FastQA reader is defined as:

```
input_module = FastQAInputModule(shared_resources)
model_module = fatqa_model_module(shared_resources)
output_module = XQAOutputModule(shared_resources)
```

The reader is an extractive QA model, further documented in [Weissenborn et al. 2017](https://arxiv.org/abs/1703.04816).
It was developed for the SQuAD dataset in `jtr/data/SQuAD`.


## Knowledge Graph Embedding Readers

Jack the Reader implements several Knowledge Graph Embedding algorithms, including:
- The Translating Embeddings Model (TransE) [Bordes et al. 2013](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)
- The Bilinear-Diagonal Model (DistMult) [Yang et al. 2015](https://ai2-s2-pdfs.s3.amazonaws.com/5b50/842142ee3efdba0ba31dff322136cd42554d.pdf)
- The Complex Embeddings Model (ComplEx) [Trouillon et al. 2016](http://proceedings.mlr.press/v48/trouillon16.pdf)

The readers are implemented as follows:

```python
@__kbp_reader
def MODEL_reader(shared_resources: SharedResources):
    """ Creates a simple kbp reader. """
    from jtr.tasks.kbp.models import KnowledgeGraphEmbeddingInputModule, KnowledgeGraphEmbeddingModelModule, \
        KnowledgeGraphEmbeddingOutputModule, KBPReader
    input_module = KnowledgeGraphEmbeddingInputModule(shared_resources)
    model_module = KnowledgeGraphEmbeddingModelModule(shared_resources, model_name=MODEL_NAME)
    output_module = KnowledgeGraphEmbeddingOutputModule()
    return KBPReader(shared_resources, input_module, model_module, output_module)
```

## CBOW XQA Reader

The CBOW XQA Reader is defined as:

```python
input_module = CBOWXqaInputModule(shared_resources)
model_module = cbow_xqa_model_module(shared_resources)
output_module = XQANoScoreOutputModule(shared_resources)
```

The reader is a bag of word embeddings extractive QA model, further documented in [Weissenborn et al. 2017](https://arxiv.org/abs/1703.04816).
It was developed for the SQuAD dataset in `jtr/data/SQuAD`.


## Example Reader

The example reader is defined as:

```python
input_module = SimpleMCInputModule(shared_resources)
model_module = SimpleMCModelModule(shared_resources)
output_module = SimpleMCOutputModule()
```

The reader is a multiple-choice QA model for questions, multiple supports and global candidates, that represents them as their averaged bag of word embeddings. Question and support representations are concatenated and are scored against a global set of candidates by taking the dot product between the question-support and the candidate representations.


## Model-F Reader

The model F reader is defined as:

```python
input_module = ModelFInputModule(shared_resources)
model_module = ModelFModelModule(shared_resources)
output_module = ModelFOutputModule()
```

The reader is a multiple-choice model for knowledge base population, described in [Riedel et al. 2013](www.aclweb.org/anthology/N13-1008) that uses questions and global candidates and no supports. Each question and candidate is viewed as one word, for which a representation is learned. Questions are scored against a global set of candidates by taking the dot product between the question and candidate representations.
The reader is developed for the NYT dataset in `jtr/data/NYT`.
