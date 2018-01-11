# Encoder Modules

We support modular composition of models for QA and NLI. This means that you can stick together your models
in a config, *no need to touch code at all*. Browse through the configs in `conf/` to get a feeling (note, some
models are implemented directly in code). Each module is defined in the config by a [yaml](http://yaml.org/) block
with predefined, module-specific properties.
The most general are listed in the following and the more specific ones further down:

* `module`: type of this module
* `name`: optional, name of the module; modules with the same name share parameters
* `input`: must be a string for sequence encoders and interaction modules, but can be a list for combination modules
* `output`: string, same as `input` by default if it is a string (not a list)
* `dropout`: if set to something bigger than 0, applies dropout after this module, (default: 0)
* `num_layers`: repeats module `num_layers` times (default: 1)
* `residual`: if `True`, residual connection of output from this module with its input (default: `False`)
* `activation`: activation function of module (`relu`, `sigmoid`, `tanh`, etc.). `identity` if set to `null` (default)

# Module Inputs and Outputs Encoders

All modules have inputs and an output referred to by a string key. The initial set of available keys is task dependent.
Note that the output of a module will become the value of a `output` key (which is the `input` by default)

### QA
There are 5 starting keys `support` (embedded support), `question` (embedded question),
`char_support` (character embedded support), `char_question` (character embedded question),
`word_in_question` (feature indicating that a word occurred in the question)

### NLI
There are 4 starting keys `premise` (embedded premise), `hypothesis` (embedded hypothesis),
`char_support` (character embedded premise), `char_hypothesis` (character embedded hypothesis).
Character based keys are only available if 'with_char_embeddings' flag is set to true in the config.

# Sequence Encoders

### BiRNNs 
* `module`: `lstm`, `gru`, `sru` (simple recurrent unit)
* `with_projection`: boolean; employs a linear projection after BiRNNs which is initialized to being the sum of the forward and backward state
* `activation`: applicable if projection is used

### Convolutions
* `module`: `conv`, `conv_glu` ([gated linear unit](https://arxiv.org/pdf/1612.08083.pdf)), 'gldr' ([gated linear dilated residual network](https://openreview.net/pdf?id=HJRV1ZZAW))
* `conv_width`: width of the convolution
* `activation`: can be set to anything (only applicable if `conv` is chosen as module) 
* `dilations`: list of dilations corresponding to the number of layers in `gldr`

### Self Attention
* `module`: `self_attn`
* `attn_type`: `mlp`, `bilinear`, `diagonal_bilinear`, `dot`
* `scaled`: if `True` use scale attention score by sqrt of dimensionality of states used for attention
* `with_sentinel`: if `True` use sentinel score in addition to attention scores before computing the softmax. Allows attending to nothing.
 
### Misc
* `module`: `dense`, `highway`,
* `activation`: applicable to all these

# Combination Modules
* `input`: list of keys to combine
* `output`: string, required
* `module`: `concat`, `add`, `sub`, `mul`, `weighted_add` (computes an element-wise sigmoid gate g; result is g * input1 + (1-g) * input2)

# Interaction Modules
These modules typically concatenate the input with some interaction states. These are usually computed using attention.

* `input`: key referring to sequence for which to compute interactions
* `dependent`: key of the interaction partner
* `concat`: concatenates input with output of this module (default: True)
* `modules`: `attention_matching`, `bidaf`, `coattention`
* `attn_type`: `mlp`, `bilinear`, `diagonal_bilinear`, `dot`
* `scaled`: if `True` use scale attention score by sqrt of dimensionality of states used for attention
* `with_sentinel`: if `True` use sentinel score in addition to attention scores before computing the softmax. Allows attending to nothing.
