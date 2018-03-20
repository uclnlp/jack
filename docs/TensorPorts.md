# TensorPorts

## Introduction
The functional interfaces between modules in jack are represented and implemented by so called TensorPorts. Much like,
TensorFlow placeholders, they are fully defined by a data-type and a corresponding shape, i.e., a known dimensionality 
of dimensions of potentially unknown sizes. Thus, they offer the functionality for creating TF placeholders when needed. 
TensorPorts are responsible for defining outputs and expected inputs of jack modules and can thus be used to validate 
their compatibility when combined to form a `JTReader`.

## Pre-defined Ports

Jack offers a number of pre-defined ports in [tensorport.py](/jack/core/tensorport.py) (see below for a code snippet) that cover specific QA application scenarios. 
They include ports for multiple-choice, extractive and generative QA settings. 

Depending on the actual task and setting, different ways of treating the input can be imagined. In case of multiple-choice QA with a fixed number C of 
candidates, for instance, we would feed candidates for a batch of examples (of size B) via a `[B, C]` tensor.
Pre-defined ports for such cases can be found in the **Ports** class of `tensorport.py`. However, this approach would not work 
anymore in case there is a varying number of candidates for each QA instance in the batch. Modeling such a setting is
possible by simply feeding a *flat* list of candidates in form of a tensor `[SUM(C_i)]` to the model while additionally 
providing a candidate-to-instance mapping *m* of shape `[SUM(C_i)]` that maps each of the candidates to the question 
(index) they belong to. Although using such Ports is the more general way of treating input it 
entails a bit of overhead which the user might want to avoid. However, depending on the application, types of Ports can of course 
be mixed, for instance, a QA setting might have a variable number of candidates while always having a fixed number of supporting texts. 

Another categorization that is present in the current implementation is the the division of ports into `Ports.Input`,
`Ports.Prediction` and `Ports.Targets`. These group ports by their application and stage within the
`JTReader` information flow. Input ports define typical inputs 
to a ModelModule within a QA setting while prediction ports should be used to define the output of a ModelModule. 
The target ports are used as typically used as `training_input_ports` of ModelModule or `training_ports` of
the InputModule. These should only be provided during training and are not necessary during evaluation or application.
Typically `training_output_ports` consists merely of `Ports.loss`, but they are not restricted to it.


## Conventions

From a user perspective it is advised to reuse as many pre-existing ports as possible. This allows for maximum
re-usability of individual modules (and Hooks). However, users are not bound to existing ports and can always define their 
own which, in some cases, will even be necessary.


## Example Ports

```python
from jack.core.tensorport import Tensorport
import numpy as np
question = TensorPort(np.int32, [None, None], "question",
                      "Represents questions using symbol vectors",
                      "[batch_size, max_num_question_tokens]")

multiple_support = TensorPort(np.int32, [None, None, None], "multiple_support",
                              ("Represents instances with multiple support documents",
                               " or single instances with extra dimension set to 1"),
                              "[batch_size, max_num_support, max_num_tokens]")

```
