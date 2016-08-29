from quebap.projects.autoread.wikireading.sampler import ContextBatchSampler as WikireadingSampler

known_datasets = ["wikireading"]

def sampler_for(dataset="wikireading"):
    assert dataset in known_datasets, "unknown dataset %s" % dataset
    return WikireadingSampler