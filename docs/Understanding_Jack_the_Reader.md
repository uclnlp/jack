# Understanding Jack the Reader

## Purpose of Jack
Jack is a library for machine reading tasks, especially questions answering, knowledge base population and recognising textual entailment, which is designed to generalize dataset and model structure. The purpose is that if one adds a new model one can test it on a range of question answering tasks, and vice versa, if one has a new dataset one can use it with all the models available in Jack. Thus with Jack we hope to push the breadth and depth of research: You design a new model and other researchers are immediately able to compare against it. If you design a new dataset you immediately have a range of baseline from the models used in jtr. Other profit from your efforts and others profit from yours.

## A high Level Overview of Jack

### Functional interfaces via TensorPorts

The main design of Jack revolves around functional interfaces between the three main modules: InputModule, ModelModule, OutputModule. The functional interface is implemented by so called [TensorPorts](TensorPorts.md) which are TensorFlow tensors wrapped in a layer of description and documentation. Each of the three modules have both input and output TensorPorts and the idea of this functional interface is that just like in functional programming languages, like Haskell, where you often can tell if a function is correct or not simply by looking at the input and output types; similarly, we can implement the same behavior in python and thus ensures correct behavior most of the time. Thus in Jack, the input and output ports must (at least partially) match between the interfaces of the three different modules. The inputs, however, are aggregated over the sequence Input -> Model -> Output such that the input interface for the output module is satisfied if all its TensorPorts occur sometime before, that is somewhere as output TensorPorts of either the InputModule or the ModelModule. As a rule this could be expressed as: **"Module: Do my inputs occur as output in some previous computation?"** In code we express this like this ([core.py](jtr/jack/core.py#L679):
```
assert all(port in self.input_module.output_ports for port in self.model_module.input_ports), \
    "Input Module outputs must include model module inputs"

assert all(port in self.input_module.training_ports or port in self.model_module.output_ports or
         port in self.input_module.output_ports for port in self.model_module.training_input_ports), \
  "Input Module (training) outputs and model module outputs must include model module training inputs"


assert all(port in self.model_module.output_ports or port in self.input_module.output_ports
         for port in self.output_module.input_ports), \
  "Module model output must match output module inputs"
```

This design introduces more boilerplate in each of the modules and can make extending modules and debugging functional interfaces cumbersome, but it ensures full integrity of the system and its modules, thus making it easier to test, and thus facilitate more aggressive refactoring which keeps the base of Jack more adaptable for future change. It also makes it obvious which Input, Model, and Output modules are interchangeable. If the interface is the same, we can exchange modules easily. Thus, if you have two models which both takes two sequences, their length, to then predict an answer from global set of candidates, you can be sure either model can be used for any Input and Output modules where the other model can be used, since their functional interface is the same.

The 3 modules are finally combined in what we call a *reader* which is an instance of the JTReader class. A reader encapsulates most of the functionality needed by a user, like saving, loading, training, processing QA pairs, and thus hides the more modular components of jack. 

### 3+1: The Three Types of Modules (Plus One)
We have the following modules with the following functionality, defined in [jtr/jack/core.py](jtr/jack/core.py):
- InputModule: is responsible for pre-processing datasets that are passed in form of a sequence of question settings (comprising question, id, support(s), answer candidates, etc.) and optional answers (depending on the functionality used). The pre-processing results typically in a mapping from tensor ports to tensors (feed-dict), or an iterator of feed-dicts, which can be passed to the subsequent ModelModule
- ModelModule: Takes TensorFlow inputs (usually word indices for word embeddings), transfers it to the GPU, runs a more or less complex model (from logistic regression to dual bidirectional LSTMs over question and support with word by word attention) to then produce some outputs
- OutputModule: This takes numpy arrays as input as generated from the ModelModule to perform complex output, such as beam search for text generation, computing top 10/100/1000 retrieval scores and more
- Hooks: These take any input from the previous Modules and compute a metric on the dataset that is passed to them. For example to compute the F1 score for the development or validation set. Hooks can also be used to generated plots from these data, or to save a model based on its validation score.
- JTReader: encapsulates most functionality and knows how to combine different modules, i.e., users only need to define modules and JTReader does the rest.

### Understanding the functional interfaces of the modules
- **Input Module**:
  - `dataset_generator(..., is_eval: bool)`: This method takes raw text input data as Q/A tuples and outputs a generator that creates batches of the tensors in the `output_ports()` and `training_ports`. The flag `is_eval` indicates if a training or validation set is passed into the method (a validation set has no `training_ports()` and its preprocessing steps may differ)
  - `setup_from_data()`: We want to use the same preprocessing whenever we call the `dataset_generator()`, for example, we want to use the same vocabularies every time we call `dataset_generator()`, thus we could want to setup a global vocabulary which is valid for all calls to `dataset_generator()`. This is exactly what this method, `setup_from_data()` is supposed to do. We can set up vocabularies, candidates for our training targets (labels from vocabulary), etc.
  - `output_ports()` output tensors generated from the InputModule which are needed to make a prediction (usually no labels required)
  - `training_ports()`  **additional** output tensors needed to generate loss value (usually only the labels here)
  - `__call__`: Used to preprocess single instances of data when using the model on totally new data (not the test data, but some new manually generated data)
  - `setup()`: is called after it is certain that the InputModule is completely configured to set up necessary resources given the configuration. Configuration might not always be existent during creation, for instance, in case of loading a saved InputModule which requires creating it first. This method is called instead of `setup_from_data()` but never together.
  
- **Model Module**:
  - This module is usually not used, but instead just serves as a baseclass for the SimpleModelModule
- **Simple Model Module**:
  - `input_ports`: define all the inputs needed for prediction
  - `output_ports`: define the TensorPorts that match the outputs generated by the `create_output()` method.
  - `training_input_ports`: define the extra Ports needed with respect to the InputPorts and that are needed for loss calculation.
  - `training_output_ports`: define the TensorPorts that match the outputs generated by the `create_training_outputs()` method.
  - `create_output()`: takes the output tensors specified by `input_ports()`, defines a model over the input and creates predictions, e.g. prediction scores or prediction labels, which can later be used by the OutputModule to produce an answer or by the `create_training_outputs()` method which is typically responsible for computing training related tensors such as the loss.
  - `create_training_outputs()`: takes outputs as defined in the `training_input_ports()` in the InputModule, as well as the `output_ports()` of the ModelModule (which is essentially the output from the `create_output()` method) and then generates a loss
- **Output Module**:
  - Implements special output processing. During application they can be used, for instance, to produce the actual outputs/answers (typically as strings) given the abstract predictions as tensors.

### How to implement functional interfaces
1. Define your input, model and output modules in the respective task file; for example if you do classification for a fixed number of classes, then add your modules into the multiple choice (mc) task file
2. You can leave the output module empty for now, as it is only needed for special processing
3. Your ModelModule can inherit from the SimpleModelModule which is simpler to implement
4. Implement the functional interfaces for each module; that is define the input and output ports. Do this by importing from Ports if your tensor are of fixed dimensions and from FlatPorts if your tensor has varying dimension (sometimes 2 dimensional and other times 3 dimensional; if the time dimension changes, still use the normal Ports)
5. Use Ports in their respective category. For example `Ports.Inputs.candidate_idx` for the index
6. Define any ports that are missing in [jtr/jack/core.py](jtr/jack/core.py)

##### Implementing the Input module

1. Implement the `setup_from_data()` method. 
  - To make things easier you can use the predefined `pipeline()` methods that do tokenization, and vocabulary creating etc. For a more detailed overview of the components of this preprocessing pipeline see the section **Low-level Preprocessing Component Description** at the end of this file
  - Save your vocabulary to a member variable so that it can be accessed during `dataset_generator()`. 
  - Make sure your method has the correct preprocessing behavior for both training set and development set, that is save the relevant vocabularies relevant for both training and test set. Note that order of labels can be different from train and dev set, thus save the word2index mappings for your labels. This is a common error during this step.
2. Implement the `dataset_generator()` method:
  - To make things easier you can use the predefined `get_batches()` method along the `pipeline()` method whenever this fits your setting.
  - make sure your implementation has the correct behavior for both the train and dev set (`is_eval=True` means dev set data is passed into the method, so condition on that)

##### Implementing the Simple Model Module

1. Overwrite the `input_ports()`, `output_ports()` and `training_output()` properties so that output from the `create_output()` method is reused in the `create_training_outputs()` 
2. extend the functional interface from a list, to spell out the actual tensor names, for example:
  ```
      # this is the template
      @abstractmethod
      def create_output(self, shared_resources: SharedResources,
                      *input_tensors: tf.Tensor) -> Sequence[tf.Tensor]:
                          @abstractmethod

      # this is the spelled out template, we removed the list argument and spelled
      # out the actual components in this list. Note that the order is the same as in the
      # input_port property of the SimpleModelModule
      def create_output(self, shared_resources: SharedResources,
                      support : tf.Tensor,
                      question : tf.Tensor,
                      support_length : tf.Tensor,
                      question_length : tf.Tensor) -> Sequence[tf.Tensor]:

      @property
      def input_ports(self) -> List[TensorPort]:
          return [Ports.Input.single_support,
                  Ports.Input.question, Ports.Input.support_length,
                  Ports.Input.question_length]


  ```
3. Implement your model in `create_output()` up to the predictions, i.e., excluding training related code which is implemented in `create_training_outputs()`.
  - You can use different predefined model blocks like bidirectional LSTMs over support and question; highway networks, fully connected projection layers and so forth which you can find in [jtr/jack/tf_fun/](jtr/jack/tf_fun/)
4. We now implement the `create_training_outputs()` which basically create the loss
5. To make the model we use in `create_output()` exchangeable, we can abstract the forward pass with another interface. This is demonstrated via the combination of (1) [`AbstractSingleSupportFixedClassModel`](jtr/jack/tasks/mcqa/abstract_multiplechoice.py), (2) [`SingleSupportFixedClassForward`](jtr/jack/tasks/mcqa/abstract_multiplechoice.py) and (3)[`PairOfBiLSTMOverSupportAndQuestionModel`](jtr/jack/tasks/mcqa/simple_mcqa.py):
    - (1) Implements everything needed for the `SimpleModelModule` interface, but abstracts the forward pass into an abstract forward method as defined by (2)
    - Thus an instance of (1) needs to implement (2)
    - (3) inherits from (1) and thus implements the model, which in this case is a pair of bidirectional LSTMs over question and support. 
    - To define another model for the same task we do not need to write a new `SimpleModelModule`, but we can inherit from (1) and just implement the forward pass for that new model
6. If you do not want your model to generalize, just skip 5., it is not mandatory. You can similarly use functional programming for additional abstraction, e.g., by defining factory methods for a certain model types which you only need to feed with existing model implementations of the respective type (see for instance [`fun.py`](jtr/jack/fun.py)). 

##### Implementing Hooks For Evaluation

1. Look at predefined hooks at [jtr/jack/train/hooks.py](jtr/jack/train/hooks.py#386) and see if you can extend them to support your metric. This depends on whether the input ports of the hook match the output ports of your modules. They are defined in the constructor.
2. Again. Make sure you define your input ports in the constructor of your hook that needs to derive from the `EvalHook` class to support plot and evaluation magic. This is the greatest source of errors when implementing a new evaluation hook
3. Define the `possible_metrics()` and `preferred_metric_and_best_score()` properties. The best score here means the score which is the lowest possible for your evaluation metric (often zero, or 0.0). This definition of best score is used to define when a model is saved after a cross validation step, that is, if the model improved or not
4. Implement your metric by implementing the method `apply_metrics()`, make sure to return a dictionary with scores for each of the metrics defined in `possible_metrics()`

##### Gluing It Together

1. Implement your reader in [jtr/jack/readers.py]('jtr/jack/readers.py'). Note that the methods with double underscore are decorators for reader functions which introduce some general behavior such
   as setting the default evaluation metric (or rather default evaluation hook) and setting the default reader that is instantiated. Your implementation of your reader should look like the `example_reader()` method


### How to Test Jack

Jack has unit tests and integration test. You can run them by running make commands in the main directory. Run `make test` to run the unit tests, and `make overfit` to run some overfit integration test. There are also more thorough integration tests which you can run with `make smalldata` but they will take much more time to run. For more information see (How to test)[How_to_test.md].

### How to Run Models

You can run models in two different ways: (1) Run the general [/jtr/jack/train/train_reader.py](jtr/jack/train/train_reader.py) script which takes the model, its model parameters and the paths to the data files and embeddings as command line parameters; (2) create your own pipeline. With the help of utility function which also make up most of the code in (1), you can create your own pipeline fairly quickly. See (this SNLI notebook)[notebooks/SNLI.ipynb].

In general you can use (1) for quick experiments and running different kind of models quickly on the same data, that is if you need a pipeline that works in general for a dataset you want the general pipeline (1). If you want to work on a specific dataset with a specific models, or if you want to include some special preprocessing steps then (2) is the best solution. If you are working on a project, it often makes sense to use (2) just for the sake for clarity, that is having more succinct, clear code.

##### How to Use the General Pipeline

1. Specify your model with the `--model`` parameter; you can see a list of models, run `python3 jtr/jack/train/train_reader.py --help`
2. Specify your data with the `--train`, `--dev` and, optionally, `--test` parameters.
3. Add training parameters such as the representation size of your model (`--repr_dim`), and the input representation (embedding size) of your model (`--repr_dim_input`)
4. Most hooks can write TF summaries so you can follow progress with TensorBoard. Use the --tb option to define the directory to which summaries are written. 

##### How to Create Your Own Pipeline

This repeats the steps of (the SNLI example notebook)[notebooks/SNLI.ipynb], with some intermediate more general steps:

1. Load your data which is in Jack format by loading it with `jtr.jack.core.load_labelled_data()`. This will convert it to the format the your reader class expects
2. Create a config dictionary with basic parameters (or special parameters for your model): 
```
config = {"batch_size": 128, "repr_dim": hidden_dim,
          "repr_dim_input": embedding_dim, 'dropout' : 0.1}
``` 
3. Use the dictionary in (2) to create your reader. If you do not have a pretrained vocabulary, you can just pass in a new, empty Vocab class:
```
reader = readers.readers["name_of_your_reader"](Vocab(), config)
```
4. Add the loss hook and the standard reader hook:
```
hooks = []
hooks.append(LossHook(reader, iter_interval=10)) # this is the standard loss hook
hooks.append(readers.eval_hooks['name_of_your_reader'](reader, dev_set, iter_interval=25)) # this is the standard metric hook, defined for your model
```

5. Add a TensorFlow optimizer
```
import tensorflow as tf
learning_rate = 0.001
optim = tf.train.AdamOptimizer(learning_rate)
```
6. Train the reader:
```
# Lets train the reader on the CPU for 2 epochs
reader.train(optim, train_set,
             hooks=hooks, max_epochs=2,
             device='/cpu:0')
```
7. We can now plot the training results:
```

# This plots the loss
hooks[0].plot()
# This plots the F1 (macro) score and accuracy between 0 and 1
hooks[1].plot(ylim=[0.0, 1.0])
```


### Re-usable Functionality in JTR

##### Preprocessing Methods (tokenize, normalize, add sequence length)
- The pipeline method is the heaviest and most detailed processing step. This method is wrangling and preprocessing data with simple call [jtr.pipeline(..)](jtr/util/pipelines.py#L115) (!!! doesn't exist anymore) but behind this method there are several preprocessing steps:
  - [jtr.preprocess.map.deep_map](jtr/util/map.py): This is a clever method which traverses a dictionary for certain keys and transforms the values of given keys in-place in a very efficient manner. It does this by using a map function to the list of value under the given dictionary keys. It is usually used to transform a list of question strings, into a tokenized version, that is transform it into a list of question word-lists
  - [jtr.preprocess.map.deep_seq_map](jtr/util/map.py): The sister of deep_map. Also applies a function and transforms the given values under a dictionary keys in-place. The difference is that it applies this functionality on lists of lists (for example tokenized questions). With that we can use this function to do many things:
    - Words -> lower case words
    - Words -> ids (and then use these ids for indices of word embeddings; this is done with the Vocab class below)
    - Words -> get length of each sentence / sequence, that is a list of sequence lengths for the entire dataset
    - Words -> Pad words with beginning and end of sentence tag, that is
[Word1, word2, word3] -> [SOS, word1, word2, word3, EOS] would be done with deep map in this way:
`deep_seq_map(corpus, lambda xs: ["<SOS>"] + xs + ["<EOS>"], ['question'])`
  - [Class jtr.preprocess.vocab.Vocab](jtr/util/vocab.py): This class builds a vocabulary from tokens (usually words) assigns an identifier to each word and maintains this map from id to word and from word to id. This class also works together with pretrained vocabularies which are then extended through more data

##### Vocab and NeuralVocab
The (Vocab)[jtr/preprocess/vocab.py#12] and (NeuralVocab)[jtr/preprocess/vocab.py#327] classes deal with vocabulary and word embedding processing and management. Vocab saves vocabulary, manages new vocabulary with pretrained vocabulary (so that you can train new words not contained in pretrained embeddings). The NeuralVocab class is a class that holds the embedding matrix and handles index-to-embedding-tensor-conversion and out-of-vocabulary (OOV) words. There is also an option for projection layer to reduce the size of the inputs into the next layer and to normalize embeddings to unit norm.
