from quebap.projects.autoread.wikireading.sampler import ContextBatchSampler as WikireadingSampler
from quebap.projects.autoread.wikipedia.sampler import BatchSampler as WikipediaSampler

known_datasets = ["wikireading", "wikipedia"]

def sampler_for(dataset="wikireading"):
    assert dataset in known_datasets, "unknown dataset %s" % dataset
    if dataset == "wikireading":
        return WikireadingSampler
    else:
        return WikipediaSampler