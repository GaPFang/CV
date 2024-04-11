################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name   = 'sgd_da_pseudo_0.7' # name of experiment

# Model Options
model_type = 'mynet' # 'mynet' or 'resnet18'

# Learning Options
epochs     = 50           # train how many epochs
batch_size = 32           # batch size for dataloader 
use_adam   = False        # Adam or SGD optimizer
lr         = 1e-2         # learning rate
milestones = [16, 32, 45] # reduce learning rate at 'milestones' epochs
start_unlabel_epoch = 16  # start using unlabel data at 'start_unlabel_epoch' epoch
threshold_k  = 0.7        # threshold for semi-supervised learning
