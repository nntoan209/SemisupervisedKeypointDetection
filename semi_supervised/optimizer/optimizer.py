from torch.optim import Adam, AdamW
import torch.nn as nn

OPTIMIZER = {
    'adam': Adam,
    'adamw': AdamW
}

def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.layers'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1

def build_optimizer(optimizer_cfg: dict, model: nn.Module):
    if hasattr(model, 'module'):
        model = model.module
        
    type = optimizer_cfg['type'].lower()
    optimizer_cfg.pop('type')
    
    paramwise_cfg = optimizer_cfg.get('paramwise_cfg', None)
    
    if paramwise_cfg is not None:
        optimizer_cfg.pop('paramwise_cfg')
        params = []
        parameter_groups = {}
        num_layers = paramwise_cfg.get('num_layers') + 2
        layer_decay_rate = paramwise_cfg.get('layer_decay_rate')
        
        lr = optimizer_cfg['lr']
        weight_decay = optimizer_cfg['weight_decay']
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if (len(param.shape) == 1 or name.endswith('.bias') or 'pos_embed' in name):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay
            layer_id = get_num_layer_for_vit(name, num_layers)
            group_name = 'layer_%d_%s' % (layer_id, group_name)
            
            if group_name not in parameter_groups:
                scale = layer_decay_rate**(num_layers - layer_id - 1)
                
                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'group_name': group_name,
                    'lr': scale * lr,
                }
            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
            
        params.extend(parameter_groups.values())
        
        optimizer = OPTIMIZER[type](params=params, **optimizer_cfg)
        
    else:
        optimizer = OPTIMIZER[type](params=model.parameters(), **optimizer_cfg)
        
    return optimizer
