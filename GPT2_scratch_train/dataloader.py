import torch,tiktoken

class CustomDataLoader:
    def __init__(self, B, T, process_rank, num_processes, split,master_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank # used for multi-gpu training
        self.num_processes = num_processes # total number of gpu_s
        assert split in {'train', 'val'} # training and validation split

         # at init load tokens from disk and store them in memory
        with open('dataset/train/input.txt', 'r') as f:
            text = f.read()
        train_split = int(0.8 * len(text))
        if split in ["train"]:
            text = text[:train_split]
        else:
            text = text[train_split:]
        enc = tiktoken.get_encoding('gpt2')

        tokens = enc.encode(text)
        total_tokens = len(tokens)
        self.tokens = torch.tensor(tokens)        

        # state
        self.current_position=0
        assert len(tokens) > 0, f"no tokens found for split {split}"
        if master_process: # only print for master gpu
            print(f"loaded {total_tokens} tokens")
        self.reset()
    
    def reset(self):
        self.current_position = self.B * self.T * self.process_rank # so that each gpu get different data
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset to original position
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = B * T * self.process_rank
        return x, y