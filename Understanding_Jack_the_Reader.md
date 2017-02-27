# Understanding Jack the Reader

## Purpose of Jack
Jack is a library which is designed to generalize dataset and model structure so that if one adds a new model one can test it on a range of question answering tasks, and vice versa, if one had a new dataset one can try all the models available in Jack. Thus with Jack we hope to push the breadth and depth of research: You design a new model and other researchers are immediately able to compare against it. If you design a new dataset you immediately have a range of baseline from the models used in jtr. Other profit from your efforts and others profit from yours.

## A high Level Overview of Jack

### Functional interfaces via TensorPorts

The main design of Jack revolves around functional interfaces between the three main modules: InputModule, ModelModule, OutputModule. The functional interface is implemented by so called TensorPorts which are TensorFlow tensors wrapped in a layer of description and documentation. Each of the three modules has both input and output TensorPorts and the idea of this functional interface is that just like in functional programming languages, like Haskell, where you often can tell if a function is correct or not simply by looking at the input and output types; similarly, we can implement the same behavior in python and thus ensures correct behavior most of the time. Thus in Jack, the input and output ports must match between the interfaces of the three different modules. The inputs however, are aggregated over the sequence Input -> Model -> Output such that the input interface for the output module is satisfied if all its TensorPorts occur sometime before, that is somewhere as output TensorPorts of either the InputModule or the ModelModule. As a rule this could be expressed as: **"Module: Do my inputs occur as output in some previous computation?"** In code we express this like this ([core.py](./jtr/jack/core.py#L679):
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

This design introduces more boilerplate in each of the modules and can make extending modules and debugging functional interfaces cumbersome, but it ensures full integrity of the system and its modules, thus making it easier to test, and thus facilitate more aggressive refactoring which keeps the base of Jack more adaptable for future change. It also makes it obvious which Input, Model, and Output modules are interchangeable. If the interface is the same, we can exchange modules easily. Thus, if you have two models which both takes two sequences, their length, to then predict an answer from global candidates, you can be sure either model can be used for any Input and Output modules where the other model can be used, since their functional interface is the same.

### 3+1: The Three Types of Modules (Plus One)
We have the following modules with the following functionality:
- InputModule: Takes some file in Jack-format (JSON file containing (1) question, (2) support (optional), (3) answer) and preprocesses the data to TensorFlow CPU tensors. This step may involve tokenizing, negative sampling (creating corrupted examples), creating vocabularies, creating word embeddings (optional: using pretraining embeddings for part of the data)
- ModelModule: Takes TensorFlow inputs (usually word indices for word embeddings), transfers it to the GPU, runs a more or less complex model (from logistic regression to dual bidirectional LSTMs over question and support with word by word attention) to then produce some outputs
- OutputModule: This takes numpy arrays as input as generated from the ModelModule to perform complex output, such as beam search for text generation, computing top 10/100/1000 retrieval scores and more
- Hooks: These take any input from the previous Modules and compute a metric on the dataset that is passed to them. For example to compute the F1 score for the development or validation set. Hooks can also be used to generated plots from these data, or to save a model based on its validation score.

### Understanding the functional interfaces of the modules
- **Input Module**:
  - `dataset_generator(..., is_eval: bool)`: This method takes raw text input data as Q/A tuples and outputs a generator that creates batches of the tensors in the `output_ports()` and `training_ports`. The flag `is_eval` indicates if a training or validation set is passed into the method (a validation set has no `training_ports()` and its preprocessing steps may differ)
  - `setup_from_data()`: We want to use the same preprocessing whenever we call the `dataset_generator()`, for example, we want to use the same vocabularies every time we call `dataset_generator()` thus we want to setup a global vocabulary which is valid for all calls to `dataset_generator()`. This is exactly what this method, `setup_from_data()` is supposed to do. We setup vocabularies, candidates for our training targets (labels from vocabulary)
  - `output_ports()` output tensors generated from the InputModule which are needed to make a prediction (usually no labels required)
  - `training_ports()`  **additional** output tensors needed to generate loss value (usually only the labels here)
  - `__call__`or `my_input_module()`: Used to preprocess single instances of data when using the model on totally new data (not the test data, but some new manually generated data)
  - `setup()`: ??? Not that important, I guess
- **Model Module**:
  - This module is usually not used, but instead just serves as a baseclass for the SimpleModelModule
- **Simple Model Module**:
  - `input_ports`: This must match the `output_ports()` of the InputModule used, otherwise Model and Input Module are not compatible
  - `create_output()`: This method takes the output tensors specified in the InputModule (and also the input tensors specified as in `input_ports()`) and creates prediction and usually logits (or the outputs that feed into the loss function)
  - `output_ports`: This module describes the TensorPorts which match the outputs generated by the `create_output()` method
  - `training_input_ports`: This must match the `training_ports()` of the InputModule used, otherwise Model and Input Module are not compatible. This method may also provides additional inputs, such as the ones generated through the `create_output()` method. Usually, you want to compute logits in the `create_output()` method, define them in this method, to then use them in `create_training_outputs()`
  - `create_training_outputs()`: This function takes outputs as defined in the `training_ports()` in the InputModule, as well as the `output_ports()` of the ModelModule (which is essentially the output from the `create_output()` method) and then generates a loss
- **Output Module**:
  - Implements special output processing which is currently not really needed nor supported by abstractions

### How to implement functional interfaces
1. Define your input, model and output modules in the respective task file; for example if you do classification for a fixed number of classes, then add your modules into the multiple choice (mc) task file
2. You can leave the output module empty for now, as it is only needed for special processing
3. Your ModelModule can inherit from the SimpleModelModule which is simpler to implement
4. Implement the functional interfaces for each module; that is define the input and output ports. Do this by important from Ports if your tensor are of fixed dimensions and from FlatPorts if your tensor Have varying dimension (sometimes 2 dimensional and other times 3 dimensional; if the time dimension changes, still use the normal Ports)
5. Use Ports in their respective category. For example `Ports.Inputs.candidate_idx` for the index
6. Define any ports that are missing

##### Implementing the Input module

1. Implement the `setup_from_data()` method. 
  - To make things easier you can use the predefined `pipeline()` methods that do tokenization, and vocabulary creating etc
  - Save your vocabulary to a member variable so that it can be accessed during `dataset_generator()`. 
  - Make sure your method has the correct preprocessing behavior for both training set and development set, that is save the relevant vocabularies relevant for both training and test set. Note that order of labels can be different from train and dev set, thus save the word2index mappings for your labels. This is a common error during this step.
2. Implement the `dataset_generator()` method:
  - To make things easier you can use the predefined `get_batches()` method along the `pipeline()` method
  - make sure your implementation has the correct behavior for both the train and dev set (`is_eval=True` means dev set data is passed into the method, so condition on that)

##### Implementing the Simple Model Module

1. Overwrite the `input_ports()`, `output_ports()` and `training_output()` properties so that output from the `create_outputs()` method is reused in the `create_training_outputs()` 
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
3. Implement the forward pass in `create_output()` (up to, not including the loss; the loss is implemented in `create_training_outputs()`
  - You can use different predefined model blocks like bidirectional LSTMs over support and question; highway networks, fully connected projection layers and so forth which you can find in [jtr/jack/tf_fun/](./jtr/jack/tf_fun/)
4. We now implement the `create_training_outputs()` which basically create the loss
5. To make the model we use in `create_output()` exchangeable, we can abstract the forward pass with another interface. This is demonstrated via the combination of (1) [`AbstractSingleSupportFixedClassModel`](.jtr/jack/tasks/mcqa/abstract_multiplechoice.py), (2) [`SingleSupportFixedClassForward`](.jtr/jack/tasks/mcqa/abstract_multiplechoice.py) and (3)[`PairOfBiLSTMOverSupportAndQuestionModel`](.jtr/jack/tasks/mcqa/simple_mcqa.py):
    - (1) Implements everything needed for the `SimpleModelModule` interface, but abstracts the forward pass into an abstract forward method as defined by (2)
    - Thus an instance of (1) needs to implement (2)
    - (3) inherits from (1) and thus implements the model, which in this case is a pair of bidirectional LSTMs over question and support. 
    - To define another model for the same task we do not need to write a new `SimpleModelModule`, but we can inherit from (1) and just implement the forward pass for that new model
6. If you do not want your model to generalize, just skip 5., it is not mandatory

##### Implementing Hooks For Evaluation

1. Look at predefined hooks at [jtr/jack/train/hooks.py](.jtr/jack/train/hooks.py#386) and see if you can extend them to support your metric. This depends on the input ports of the hook match the output ports of your module. These are defined in the constructor.
2. Again. Make sure you define your input ports in the constructor of your hook that needs to derive from the `EvalHook` class to support plot and evaluation magic. This is the greatest source of errors when implementing a new evaluation hook
3. Define the `possible_metrics()` and `preferred_metric_and_best_score()` properties. The best score here means the score which is the lowest possible for your evaluation metric (often zero, or 0.0). This definition of best score is used to define when a model is saved after a cross validation step, that is, if the model improved or not
4. Implement your metric by implementing the method `apply_metrics()`, make sure to return a dictionary with scores for each of the metrics defined in `possible_metrics()`

##### Gluing It Together

1. Implement your reader in [jtr/jack/readers.py]('.jtr/jack/readers.py'). Note that the methods with double underscore are decorators for reader functions which introduce some general behavior such
   as setting the default evaluation metric (or rather default evaluation hook) and setting the default reader that is instantiated. Your implementation of your reader should look like the `example_reader()` method


# Missing:
### How to test
### How to run models (with and without pipeline)
### Description of sub-components


## A High Level Overview of jtr
jtr is best understood by going from the high-level function and classes to low-level function and classes. The highest level entry point of jtr is the [training_pipeline.py](./jtr/training_pipeline.py). The training pipeline calls other high-level functions step-by-step, from input data, over preprocessing and data wrangling, to training the selected model. Although the [training_pipeline.py](./jtr/training_pipeline.py) script is more complex, here are some other high level functions and classes which are traversed while going along the pipeline from data to model.
   The script does step-by-step:
### 1. Define jtr models
- [Predefined models](../jtr/nn/models.py) found in jtr.nn.models such as:
  - Conditional reader: Two bidirectional LSTM over two sequences where the second is conditioned on the final state of the other
  - Attentive reader: Like conditional reader, but all states are processed with attention so that the most important bits of each of the two sequences are filtered from the unimportant bits. Finally these two streams of filtered information are combined

### 2. Parse the input arguments
- Standard [argparse](https://docs.python.org/3/library/argparse.html). Arguments include: Batchsize, pretrain embeddings, learning rate, l2 penalty, clip value (gradients), dropout, epochs, negative sampling (amount)

### 3. Read the train, dev, and test data
- Uses [jtr.load.read_jtr.jtr_load](./jtr/load/read_jtr.py) which loads a JSON file in a specific jtr format. To transform your data into this jtr JSON format there exist scripts which wrangle certain data sets into the required format. You can find these data wrangling scripts under the path [jtr/jtr/load/](./jtr/load). The JSON format can be seen as a python dictionary which contains high level names for different kind of data:
  - Question (Q): Question text or binary relation like (Donald Trump, presidentOf, X)
  - Support (S): Supportive text passage for the question.
  - Candidates (C): A corpus may have 10000 entities, but for a question only 10 candidates might be relevant of which 2 are correct, for example all entities in the supporting passage are candidates. Candidates might also refer to all words in the vocabulary (no restrictions).
  - Answers: The answer or answers to the question or binary relation. This may also be a span like (17, 83) indicating the answer is located between character position 17 and 83 (Stanford SQuAD)
- At this point in the pipeline one can also load pretrained embeddings and merge them with the vocabulary of the loaded datasets

### 4. Preprocesses the data (tokenize, normalize, add  start and end of sentence tags) via the jtr.pipeline method
- This is the heaviest and most detailed processing step. In the script this data wrangling and preprocessing pipeline is called with a simple call [jtr.pipelines.pipeline(..)](../jtr/pipelines.py) but behind this method there are several preprocessing steps:
  - [jtr.preprocess.map.deep_map](../jtr/preprocess/map.py): This is a clever method which traverses a dictionary for certain keys and transforms the values of given keys in-place in a very efficient manner. It does this by using a map function to the list of value under the given dictionary keys. It is usually used to transform a list of question strings, into a tokenized version, that is transform it into a list of question word-lists
  - [jtr.preprocess.map.deep_seq_map](../jtr/preprocess/map.py): The sister of deep_map. Also applies a function and transforms the given values under a dictionary keys in-place. The difference is that it applies this functionality on lists of lists (for example tokenized questions). With that we can use this function to do many things:
    - Words -> lower case words
    - Words -> ids (and then use these ids for indices of word embeddings; this is done with the Vocab class below)
    - Words -> Pad words with beginning and end of sentence tag, that is
[Word1, word2, word3] -> [SOS, word1, word2, word3, EOS] would be done with deep map in this way:
`deep_seq_map(corpus, lambda xs: ["<SOS>"] + xs + ["<EOS>"], ['question'])`
  - [Class jtr.preprocess.vocab.Vocab](../jtr/preprocess/vocab.py): This class builds a vocabulary from tokens (usually words) assigns an identifier to each word and maintains this map from id to word and from word to id. This class also works together with pretrained vocabularies which are then extended through more data

### 5. Create NeuralVocab
- A word to embedding class which manages the training of embedding which optionally may be enriched with some pretrained embeddings. Parameters may be frozen and there are options for a projection layer to reduce the size of the inputs into the next layer and to normalize embeddings to unit norm.

### 6. Create TensorFlow placeholders and initialize model
### 7. Batch the data via jtr.preprocess.batch.get_feed_dicts
### 8. Add hooks
- [Hooks](../jtr/util/hooks.py) are functions which are invoked after either the end of an iteration or the end of an epoch. They usually print some information (loss value, time taken this epoch, ETA until the model is fully trained, statistics of certain tensors like weights) and save this information to the TensorFlow summary writer so that these data can be visualized via TensorBoard.

### 9. Train the model
- Calls [jtr.train.train(..)](../jtr/train.py) with everything which was constructed in the previous steps such as the batched data, the model, the hooks, and the parameters.
