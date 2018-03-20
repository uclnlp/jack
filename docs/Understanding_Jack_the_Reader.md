# Understanding Jack the Reader

## Purpose of Jack
Jack is a library for machine reading tasks including question answering, knowledge base population via link-prediction and recognising textual entailment. Jack's purpose in life is to allow for adding new models and testing them on a range of datasets, and vice versa, if one has a new dataset one can use it with all the models available in Jack. Thus with Jack we hope to push the breadth and depth of research: You design a new model and other researchers are immediately able to compare against it. If you design a new dataset you immediately have a range of baseline from the models used in jtr. Other profit from your efforts and you profit from others.

If you want to get your hands dirty right away, why not playing around with our [notebooks](/notebooks) to get a feeling of Jack. You can also quickly train a model or use a pre-trained model as described in the [readme](/README.md). If you like reading as much as Jack does and want to develop, just read through the rest of this document which explains Jack's core in more detail.

## A high Level Overview of Jack

### Functional interfaces via TensorPorts

The main design of Jack revolves around functional interfaces between the three main modules: InputModule, ModelModule, OutputModule. The functional interface is implemented by so called [TensorPorts](TensorPorts.md). These are tensor definitions that include at least a shape, description and name. Each of the three modules have both input and output TensorPorts and the idea of this functional interface is that just like in functional programming languages, like Haskell, where you often can tell if a function is correct or not simply by looking at the input and output types; similarly, we can implement the same behavior in python and thus ensure correct behavior most of the time. Thus in Jack, the input and output ports must (at least partially) match between the interfaces of the three different modules. The inputs, however, are aggregated over the sequence Input -> Model -> Output such that the input interface for the output module is satisfied if all its TensorPorts occur sometime before, that as output TensorPorts of either the InputModule or the ModelModule. As a rule this could be expressed as: **"Module: Do my inputs occur as output in some previous computation?"** In code we express this like this ([reader.py](/jack/core/reader.py#L45)):
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

This design introduces more boilerplate in each of the modules and can make extending modules and debugging functional interfaces cumbersome, but it ensures full integrity of the system and its modules, thus making it easier to test, and thus facilitate more aggressive refactoring which keeps the base of Jack more adaptable for future change. It also makes it obvious which Input, Model, and Output modules are interchangeable by knowing directly from the TensorPorts what they consume and produce. If the interface is the same, we can exchange modules easily. Thus, if you have two models which both takes two sequences, their length, to then predict an answer from global set of candidates, you can be sure either model can be used for any Input and Output modules where the other model can be used, since their functional interface is the same.

The 3 modules are finally combined in what we call a *reader* which is an instance of the JTReader class. A reader encapsulates most of the functionality needed by a user, like saving, loading, training, processing QA pairs, and thus hides the more modular components of jack.

### Data Structure
In jack, everything is expressed in form of QASettings and Answers (see [data_structures](/jack/core/data_structures.py)). A `QASetting` consists of at least a question, and optionally some support texts that should help to answer the question, atomic or sequential (i.e., typically token sequences) candidates for multiple choice, or candidate spans for multiple choice over the support. We believe that most tasks can easily be expressed in such a format. For instance, for natural language inference the question becomes the hypothesis and support contains a single text, namely the premise. Furthermore, it also has candidates which are the potential multiple choice classes *entailment*, *contradiction* or *neutral*. An `Answer` is only provided during training and produced during application by our OutputModule (see below). It consists of the answer as a string and potentially its span in one of the supporting texts of the corresponding `QASetting`.

### 3+1: The Three Types of Modules (Plus One)
We have the following modules with the following functionality, defined in [jack/core](/jack/core):
- InputModule: is responsible for pre-processing datasets that are passed in form of a sequence of question settings (comprising question, id, support(s), answer candidates, etc.) and optional answers (depending on the functionality used). The pre-processing results typically in a mapping from tensor ports to tensors (feed-dict), or an iterator of feed-dicts, which can be passed to the subsequent ModelModule
- ModelModule: Takes outputs of the InputModule (e.g., word indices for word embeddings), runs a more or less complex model (from logistic regression to dual bidirectional LSTMs over question and support with word by word attention) to then produce predictions and other outputs for training, like the loss.
- OutputModule: This takes numpy arrays as input generated from the ModelModule to create an output typically in human readable form. However, it can be used for all sorts of post-processing given the output of the ModelModule.
- JTReader: encapsulates most functionality and knows how to combine different modules, i.e., users only need to define modules ahnd them to JTReader and JTReader does the rest.

### Understanding the functional interfaces of the modules

- **InputModule**:
  - `batch_generator(..., is_eval: bool)`: This method takes raw text input data as Q/A tuples and outputs a generator that creates batches of the tensors defined by `output_ports` and `training_ports`. The flag `is_eval` indicates if a training or validation set is passed into the method to change the input behaviour if necessary.
  - `__call__`: Used to preprocess single instances of data when using the model for application on-the-fly.
  - `setup_from_data()`: We want to use the same preprocessing whenever we call the `batch_generator()`, for example, we want to use the same vocabularies every time we call `batch_generator()`, thus we could want to setup a global vocabulary which is valid for all calls to `batch_generator()`. This is exactly what this method, `setup_from_data()` is supposed to do. We can set up vocabularies, candidates for our training targets (labels from vocabulary), etc. Resources created here that need to be shared with another modules shuold be handed over to the SharedResources. In general we try to keep our modules stateless and let SharedResources hold all state information. `setup_from_data()` is only called just before training and should never be called anywhere else.
  - `setup()`: is called after it is certain that the provided shared resources are completely loaded or setup. Configuration might not always be existent during creation, for instance, in case of loading a saved reader which requires creating it first. This method is called instead of `setup_from_data()` but never together.
  - `output_ports()` defines the type of output tensors generated from the InputModule needed to only make a prediction (usually no labels required)
  - `training_ports()`  **additional** output tensors needed to generate loss value (usually only the labels here)

- **OnlineInputModule**:
  - This is a simplified, more intuitive version of InputModule and should be used in most cases instead. Instead of `__call__` and `batch_generator()` you need to implement the following.
  - `preprocess()`: take a list (e.g., dataset) of QuestionSettings and optionally provided Answers, preprocess them as neccesary and return an aligned list of annotations for each QuestionSetting, e.g., tokens, etc.
  - `create_batch()`: get a slice/batch of the annotations created in `preprocess`, tensorize them and create a feed dict from output tensorports to numpy arrays.

- **ModelModule**:
  - This module is usually serves as a baseclass for the TFModelModule and potentially other ModelModules in the future.
  - `input_ports`: define all the inputs needed for prediction
  - `output_ports`: define the TensorPorts that are produced by this module.
  - `training_input_ports`: define the extra Ports needed with respect to the InputPorts and that are needed for the computation of the defined `training_output_ports`.
  - `training_output_ports`: define the TensorPorts for the tensors that are produced additionally during training, usually only the loss.
- **TFModelModule**:
  - This is our tensorflow backed ModelModule.
  - `create_output()`: takes the output tensors specified by `input_ports()`, defines a model over the input and creates predictions, e.g. prediction scores or prediction labels, which can later be used by the OutputModule to produce an answer or by the `create_training_outputs()` method which is typically responsible for computing training related tensors such as the loss.
  - `create_training_outputs()`: takes outputs as defined in the `training_input_ports()` in the InputModule, as well as the `output_ports()` of the ModelModule (which is essentially the output from the `create_output()` method) and then generates a loss
- **Output Module**:
  - Implements special output processing. During application they can be used, for instance, to produce the actual outputs/answers (typically as strings) given the abstract predictions as tensors.

### How to implement a new reader
Define your input, model and output modules (browse [readers](/jack/readers/implementations.py) for examples). Input and output modules can be re-used most of the time. Why not just looking for the closest reader to the one you want to write and simply writing a new ModelModule while reusing Input- and OutputModule. Check out our [notebook](/notebooks/Implementing_a_new_model.ipynb) that guides you through this process for a simple QA model.

##### Implementing Hooks For Evaluation

1. We use hooks for evaluation during training and they need to be defined for each type of reader. Look at predefined hooks at [jtr/jack/train/hooks.py](/jtr/train/hooks.py#386) and see if yours already exists or if you can extend them to support your metric. This depends on whether the input ports of the hook match the output ports of your modules. They are defined in the constructor. For our current tasks these are defined, so nothing todo here.
2. Again. Make sure you define your input ports in the constructor of your hook that needs to derive from the `EvalHook` class to support plot and evaluation magic. This is the greatest source of errors when implementing a new evaluation hook
3. Define the `possible_metrics()` and `preferred_metric_and_initial_score()` properties. The initial score here means the score which is the lowest possible for your evaluation metric (often zero, or 0.0).
4. Implement your metric by implementing the method `apply_metrics()`, make sure to return a dictionary with scores for each of the metrics defined in `possible_metrics()`

##### Gluing It Together

Define your reader in [/jack/readers/implementations.py]('/jack/readers/implementations.py'), which is used for book-keeping all our readers. Note that the methods with double underscore are decorators for reader functions which introduce some general behavior such as setting the default evaluation metric (or rather default evaluation hook) and coupling your reader with the function name of your factory method. This name can be used for instance in a configuration and `bin/jack-train.py` will know which reader to create.

