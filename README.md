This project contains GPT2 (124 M parameters) model trained from scratch in Pytorch.

The implementation has been divided into two parts :

Implementation 1 - 

This conatins code for GPT-2 model structure and inference is performed by loading weights from huggingface directly.

GPT2_from_hf/GPT2_from_hf.py -  This contains GPT-2 model structure code with method for loading model weights from HuggingFace to perform inference.

GPT2_from_hf/output.txt - This contains sample output of inference performed using Huggingface model weights.


Implementation 2 - 

This contains code to train GPT-2 from scratch including multiple GPU training support.

GPT2_scratch_train/train_GPT2_scratch.py - It contains code for training GPT-2 model from scratch with several training 
optimizations such as compiling model,cosine learning decay, gradient accumulation mechanism for less batch size if less GPU memory, conversion to lower precision datatype, changing model parameters to nice numbers(power of 2)
for better GPU calculation optimizations, etc. including multi-GPU training support using Pytorch's DistributedDataParallel module.

GPT2_scratch_train/GPT2_scratch.py - It contains GPT2 model structure with necessary optimizer functions.

GPT2_scratch_train/dataloader.py - This contains code for Custom Dataloader for training.

GPT2_scratch_train/final_output.txt - This contains output after training GPT-2 model from scratch (less parameter count) on single collab GPU free-tier after 30 minutes of training.

This code is inspire by Andrej Karpathy.
