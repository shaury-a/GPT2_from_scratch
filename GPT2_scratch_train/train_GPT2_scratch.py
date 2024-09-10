import os
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken
from dataloader import CustomDataLoader
from GPT2_scratch import GPT,CustomConfig
import math
import time
from torch.nn import functional as F

def get_lr(step,warmup_steps,max_lr,min_lr,max_steps):#need to understand
    # 1) linear warmup for warmup_iters steps
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    # 2) if step > lr_decay_iters, return min learning rate
    if step > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def train_GPT2():
    ddp = int(os.environ.get('RANK', -1)) != -1 # flag,checking if multi-gpu run or not, if RANK not set (set by torchrun) then return -1.
    if ddp:
        # set the device appropriately according to rank
        assert torch.cuda.is_available(), "need CUDA for DDP"
        init_process_group(backend='nccl') # supports lot of features for gpu communication
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE']) # total number of gpu's available
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # finding main gpu
    else:
        # normal run, no multiple gpu_s
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device_type = "cuda" if device.startswith("cuda") else "cpu"

     #setting for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2) #need to be set exclusively for gpu
    else:
        torch.manual_seed(2)

    enc = tiktoken.get_encoding("gpt2")

    total_batch_size = 48000 #setting a nice number, in powers of 2
    B = 64 # micro batch size
    T = 1024 # sequence length
    if ddp:
        assert total_batch_size % (B * T * ddp_world_size) == 0, "making sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = CustomDataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train",master_process=master_process)
    val_loader = CustomDataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val",master_process=master_process)
    torch.set_float32_matmul_precision('high') #lowering precision, faster training

    # create model
    model = GPT(CustomConfig(vocab_size=50304)) # increased vocab size to a nice number(in power of 2) for faster training
    model.to(device)
    use_compile = False # torch.compile interferes with HellaSwag eval and Generation.
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # Implementing cosine decay learning rate
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 2
    max_steps = 6 # number of steps in 1 epoch, 6.7 steps is ~1 epoch, if data is 338025 tokens and batch size 0.05M tokens
    # optimize!
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type,master_process=master_process)

    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # evaluating after 2 steps
        if step % 2 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 2
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):# handles mixed-precision
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach() #detaching loss from pytorch computational graph, no gradients required
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG) # perform avg operation across all gpu_s
            if master_process:# only main gpu will do the logging
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 2 == 0 or last_step):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    # you might also want to add optimizer.state_dict() and
                    # rng seeds etc., if you wanted to more exactly resume training
                    torch.save(checkpoint, checkpoint_path)

        # once in a while generate from the model (except step 0, which is noise)
        if ((step > 0 and step % 2 == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("What is so great about India is,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device) # done so not to disturb train random seed
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen) # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :] # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # perform grad sync on last step only
            
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step,warmup_steps,max_lr,min_lr,max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    train_GPT2()
