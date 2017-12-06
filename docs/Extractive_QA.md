# Extractive Question Answering

Extractive or document question answering aims at finding a span `(document_index, start, end)` representing the answer
to a given question. It assumes that such documents are given apriori. We call supporting documents simply the "support"
in our framework.

### Pre-trained Models

We list trained models with scores, relative speeds and number of parameters for models.
 
#### SQuAD (Devset results)

| Model         |  F1  | Exact  | Speed | Params in M  | Download        |
|---------------|------|--------|-------|--------------|-----------------|
| BiDAF         | 77.8 |  68.5  |  1.0  |     2.02     | [bidaf]         |
| FastQA        | 77.4 |  67.4  |  2.2  |     0.95     | [fastqa]        |
| JackQA Light  | 79.5 |  69.7  |  2.0  |     1.17     | [jackqa light]  |
| JackQA        | TBD  |  TBD   |  1.5  |     2.24     | [jackqa]        |


[bidaf]: https://www.dropbox.com/s/63zkes18juan7wa/bidaf.zip?dl=1
[fastqa]: https://www.dropbox.com/s/qb796uljoqj0lvo/fastqa.zip?dl=1
[jackqa light]: https://www.dropbox.com/s/061y88hcq62az63/jack_qa_light.zip?dl=1
[jackqa]: https://www.dropbox.com/s/36b6tju9engorgc/jack_qa.zip?dl=1

### Implementing new Models

Please make sure that you roughly [understand Jack](/docs/Understanding_Jack_the_Reader.md) before going on with
this document.

New models can be stuck together using only the configuration file. All our model implementations except FastQA are
defined using our `ModularQAModel`. Possible modules are explained in more detail [here](/docs/Encoder_Modules.md).
Example configs can be found in `conf/qa`. However, if you think your fancy model cannot be defined like this please
keep on reading.

In preparation it is advised to at least go through the
[Implementing_a_new_model.ipynb](/notebooks/Implementing_a_new_model.ipynb) to understand the basics behind Jacks modular
design and how to use it to implement new models.

Jack contains implementations for extractive question answering in the [extractive_qa](/jack/readers/extractive_qa)
package. They include reusable modules defined in [extractive_qa/shared.py](/jack/readers/extractive_qa/shared.py),
e.g., `XQAInputModule`, `XQAOutputModule` and an `AbstractXQAModelModule`. This makes it very easy to implement new
models, because all you need to implement is a single function, see for instance [FastQA](/jack/readers/extractive_qa/fastqa.py). 
After implementing your model you merely need to register it in 
[implementations.py](/jack/readers/implementations.py). After that you are ready to train your model using the function
name (`new_xqa_reader`) you specified in the `implementations.py` file as value for the `reader` flag in you
configuration. You can of course reuse existing configs (e.g., `fastqa.yaml`) and overwrite the `reader` flag on the
command line, e.g., with the following command:

```bash
$ python3 bin/jack-train.py with config='./conf/fastqa.yaml' reader=new_xqa_reader
```

### Extractive QA InputModule Implementation

Our existing input module serves a bunch of potentially useful TensorPorts, but of course not all of them need to be
used. Take a look at its defined `output_ports` and the existing model implementations for reference.
It also supports multi-paragraph/-document QA (we call supporting paragraphs/documents simply "support"). 
That means, that there might be multiple supports for a single question. The alignment between question and corresponding
supports is represented by the `support2question` tensor. If there are too many supporting documents for a single
question, that is, the `max_num_support` configuration is set to something lower, then the input module will sub-sample
paragraphs using TF-IDF similarity between question and support and only retain the `max_num_support` highest ranked.

If the XQAInputModule doesn't provide a crucial "tensor" for your new model, you can of course simply implement your own
InputModule.

### Supported Models

By the time of writing, Jack supports the following implementations: [FastQA][fastqa], [BiDAF][bidaf] and [DCN+][dcn+],
as well ass our own creation JackQA, which aims at being resource friendly while retaining good performance. 
Ready-to-use configurations for training such models can be found in the `conf/qa` directory.

[fastqa]: https://arxiv.org/abs/1703.04816
[bidaf]: https://arxiv.org/abs/1611.01603
[dcn+]: https://arxiv.org/abs/1711.00106

### Supported Datasets

There are download and conversion scripts for SQuAD, TriviaQA and NewsQA in `data/`. NewsQA can be converted to either
`jack` or `squad` format after download using the converters in `jack/io`. Setting up TriviaQA needs a couple of hours 
but is fully automatized by running the download script resulting in `jack` formatted datasets. See the 
`data/triviaqa/README` for more details. Supported loaders include `'jack'` and `'squad'`, so no need to convert your
dataset if it comes in `squad` format.


[squad]: https://rajpurkar.github.io/SQuAD-explorer/
[triviaqa]: http://nlp.cs.washington.edu/triviaqa/
[newsqa]: https://datasets.maluuba.com/NewsQA
