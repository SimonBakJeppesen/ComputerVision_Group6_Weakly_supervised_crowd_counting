import torch
import shutil
import numpy as np
import random
            
def save_checkpoint(state,is_best, task_id, filename='checkpoint.pth', fold=''):    
    if is_best:
        torch.save(state, './'+str(task_id)+'/'+fold+'model_best.pth') #filename) #Move this line outside if for original code
        #shutil.copyfile('./'+str(task_id)+'/'+filename, './'+str(task_id)+'/'+fold+'model_best.pth')


def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

