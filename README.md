# Automatic Batch Size Function
This repository aims to create a function that can be used to find the optimal batch size to use before training a model. The other files in this repository allow for the script to be run on the [Boston University's Shared Computing Cluster (SCC)](https://www.bu.edu/tech/support/research/computing-resources/scc/), the batch system of which is based on the [Sun Grid Engine](https://gridscheduler.sourceforge.net/) (SGE) scheduler.

## The optimal_batch_size function
Within batch.py the following function can be seen:
```
# Function to find optimal batch size
def optimal_batch_size(dataset, model, criterion, optimizer, starting_batch_size=64):
    batch_size = starting_batch_size
    lower_bound, upper_bound = 0, None

    while True:
        print(f"Trying to run epoch with batch size: {batch_size} @ {datetime.datetime.now()}")
        try:
            # Attempt train
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            train(dataloader, model, criterion, optimizer, epochs=1) 

            lower_bound = batch_size  # If successful, set lower bound to current batch_size
            if upper_bound is None:  # If doubling has not failed yet, double batch size again
                batch_size *= 2
            else:  
                prev = batch_size
                batch_size = (lower_bound + upper_bound) // 2    # Binary search algorithm to find optimal batch size
                if prev == batch_size:
                    return batch_size           # If batch size doesn't change then return final batch size

        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Memory error: {e}")
                upper_bound = batch_size  # If fail, set upper bound to current batch size
                batch_size = (lower_bound + upper_bound) // 2
                if upper_bound - lower_bound <= 1:  
                    return lower_bound      # Return optimal batch size
            else:
                raise e     # Real error
```
The way this function works is that it attempts to train the model for only one epoch with each batch size. Should the training fail due to a 'CUDA out of memory' error, the batch size will be altered accordingly. If another error occurs, the code will stop and the real error will be executed. The algorithm used to change the batch size is the binary search algorithm, meaning that the batch size will keep doubling until fail and then go half way down to the last successful batch size.

Given the nature of this function, another function that makes training with one epoch possible is a requirement. For reference, my train function looks like this:
```
def train(dataloader, model, criterion, optimizer, epochs):
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for images in dataloader:
            images = images.cuda(non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            labels = torch.randint(0, 10, (images.size(0),)).cuda(non_blocking=True)    # Fake labels
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch complete")
```

## Example of function usage
A log reflecting a successful usage of this function is provided at `./logs/success.qlog`
I had starting_batch_size = 64 for this example. Here is a little snippet of the log that reflects what happens upon encountering a GPU memory error:
```
Trying to run epoch with batch size: 1024 @ 2024-07-31 16:57:19.598383
Epoch complete
Trying to run epoch with batch size: 2048 @ 2024-07-31 16:57:27.390926
Memory error: CUDA out of memory. Tried to allocate 98.00 MiB. GPU 
Trying to run epoch with batch size: 1536 @ 2024-07-31 16:57:33.068148
Epoch complete
```
A batch size of 1024 worked for this model, but 2048 was too high, resulting in the next batch size tested being 1024+(2048-1024)/2 = 1536. 

Hopefully, this function can be seamlessly integrated into your models and help with optimizing training. If you have any further questions or spot any errors, please contact at me at rsyed@bu.edu.
