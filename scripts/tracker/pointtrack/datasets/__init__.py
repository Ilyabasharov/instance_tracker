from datasets.KittiMOTSDataset import *


def get_dataset(name, dataset_opts):
    if name == "mots_track_train":
        return MOTSTrackCarsTrain(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))