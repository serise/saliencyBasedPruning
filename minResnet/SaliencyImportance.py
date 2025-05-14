import torch
import cv2
import math
import gc
from utils.trainingUtils import load_model

class SaliencyImportance:
    def __init__(self, config):
        self.pruning_rate = config['pruning']['pruning_rate']
        self.step_names = config['pruning']['step_names']
        self.prune_list={}
        self.use_gpu = config['fine_tuning']['use_gpu']
        self.checkpoint_path = config['model']['checkpoint_path']
        self.num_classes = config['model']['num_classes']
        self.modelName = config['prune_setting']['model_name']
        self.model = None

    def load_model(self):
        self.model = load_model(self.modelName,numClasses=self.num_classes)
        self.model.load_state_dict(torch.load(self.checkpoint_path))

    def calculate_saliency_importance(self, output):
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        cpu_output = output.cpu()
        importances = torch.zeros(output.shape[1])

        for j in range(output.shape[1]):
            for i in range(output.shape[0]):
                (success, saliency_map) = saliency.computeSaliency(cpu_output[i, j, :, :].detach().numpy())
                importances[j] += (saliency_map>0.2).sum()

        return importances

    def forward_saliency_hook(self, prune_rate, name):
        def hook_function(module, input, output):
            importance = self.calculate_saliency_importance(output)

            topk = math.floor(len(importance) * prune_rate)
            ltopk = len(importance) - topk
            ltopk_values, ltopk_indices = torch.topk(importance, k=ltopk,largest=False)
            ltopk_indices_list = ltopk_indices.tolist()
            self.prune_list[name] = ltopk_indices_list

        return hook_function

    def analyze_model_saliency(self, data_loader):
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hook_fn = self.forward_saliency_hook(self.pruning_rate, name)
                hooks.append(module.register_forward_hook(hook_fn))

        input_data = next(iter(data_loader))[0]
        if self.use_gpu:
            input_data = input_data.cuda()
            self.model = self.model.cuda()
        _ = self.model(input_data)

        for hook in hooks:
            hook.remove()

    def clear_cache(self):
        if self.model is not None:
            if next(self.model.parameters()).is_cuda:
                self.model = self.model.cpu()
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_prune_list(self):
        return self.prune_list

