import torch
import copy
from typing import Dict

def get_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

class TaskArithmeticSurgeon:
    @staticmethod
    def compute_task_vector(base_state: Dict[str, torch.Tensor],
                             finetuned_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        task_vector = {}
        for key in base_state.keys():
            if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                continue
            if key in finetuned_state:
                task_vector[key] = finetuned_state[key] - base_state[key]
        return task_vector

    @staticmethod
    def mask_task_vector(task_vector, drop_percentile):
        #pruning con subsampling per evitare problemi di memoria
        if drop_percentile <= 0:
            return task_vector
            
        new_vector = {}
        for name, param in task_vector.items():
            if param.numel() <= 1:
                new_vector[name] = param
                continue
                
            flat_param = torch.abs(param.view(-1).cpu().float())
            
            MAX_ELEMENTS = 5_000_000 # limite per pytorch
            
            if flat_param.numel() > MAX_ELEMENTS:
                indices = torch.randint(0, flat_param.numel(), (MAX_ELEMENTS,))
                sample = flat_param[indices]
                threshold = torch.quantile(sample, drop_percentile)
            else:
                threshold = torch.quantile(flat_param, drop_percentile)
            
            mask = (torch.abs(param) >= threshold.to(param.device)).float()
            new_vector[name] = param * mask
            
        return new_vector

    @staticmethod
    def apply_task_vector(base_state: Dict[str, torch.Tensor],
                          task_vector: Dict[str, torch.Tensor],
                          alpha: float = 1.0) -> Dict[str, torch.Tensor]:
        #Applica il task vector al base state, theta_unlearned = theta_base + alpha * tau_forget
        new_state_dict = {}
        for key in base_state.keys():
            if key in task_vector:
                new_state_dict[key] = base_state[key] + alpha * task_vector[key]
            else:
                new_state_dict[key] = base_state[key]
        return new_state_dict
        
    @staticmethod
    def unlearn(base_model: torch.nn.Module, 
                forget_expert_model: torch.nn.Module, 
                alpha: float = -1.0,
                drop_percentile: float = 0.0) -> torch.nn.Module: # alpha < 0 per rimuovere e alpha > 0 per aggiungere
        unlearned_model = copy.deepcopy(base_model)
        base_sd = get_state_dict(base_model)
        forget_sd = get_state_dict(forget_expert_model)
        tau_forget = TaskArithmeticSurgeon.compute_task_vector(base_sd, forget_sd)
        if drop_percentile > 0.0:
            tau_forget = TaskArithmeticSurgeon.mask_task_vector(tau_forget, drop_percentile)
        unlearned_sd = TaskArithmeticSurgeon.apply_task_vector(base_sd, tau_forget, alpha=alpha)
        unlearned_model.load_state_dict(unlearned_sd)
        return unlearned_model
