import torch
import torch_pruning as tp
from SaliencyImportance import SaliencyImportance
from Pruner import Pruner
from utils.trainingUtils import *


def main():
    config = load_config('config.json')

    modelName = config['prune_setting']['model_name']
    datasetName = config['prune_setting']['dataset_name']
    mean = config['dataset'][datasetName]['train_mean']
    std = config['dataset'][datasetName]['train_std']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    num_classes = config['model']['num_classes']
    checkpoint_path = config['model']['checkpoint_path']

    training_loader = get_training_dataloader(
        datasetName,mean,std,num_workers=num_workers,batch_size=batch_size,shuffle=True
    )
    test_loader = get_test_dataloader(
        datasetName, mean, std,num_workers=num_workers,batch_size=batch_size,shuffle=False
    )

    saliencyImportance = SaliencyImportance(config)
    saliencyImportance.load_model()
    saliencyImportance.analyze_model_saliency(training_loader)
    saliencyImportance.clear_cache()
    prune_list = saliencyImportance.get_prune_list()

    model = load_model(modelName,numClasses=num_classes)
    model.load_state_dict(torch.load(checkpoint_path))

    pruner = Pruner(config,model,training_loader,test_loader,prune_list)
    pruner.prune()
    pruner.printFlopsAndParams()


if __name__ == "__main__":
    main()
