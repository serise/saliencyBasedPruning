# saliencyBasedPruning

## Usage

1.Put the pretrained model in ./models/checkpoint

2.Edit the config file in ./minResnet, set model_name, dataset_name, num_classes, checkpoint_path. model_name currently supports ['resnet32','resnet56','resnet110','vgg16']
and dataset_name currently supports ['cifar10','cifar100'].

3.run python ./minResnet/main.py to run the code.
