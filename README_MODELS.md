# Model Changes

20221114 The model runs on GC with the trainingModel wrapper.

From Alex T

I suggest the following steps to port the model, assuming it is PyTorch for training and inference:

1. (Done) Make sure you can run it on CPU
2. (Done) Set batch size to 1
3. (Done) Create a training model with loss, and wrap it in poptorch.trainingModel, see [documentation](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/overview.html#poptorch-trainingmodel) and [tutorial](https://github.com/graphcore/tutorials/tree/master/tutorials/pytorch/basics#build-the-model) - this line seems a good place to start from: [trainer.py](https://github.com/coreyjadams/CosmicTagger/blob/master/src/utils/torch/trainer.py#L742)
4. Wrap the inference model in  poptorch.inferenceModel (link)
5. Run to see if it fits in memory
6. If it does not: create a profile with the [Graph Analyser](https://docs.graphcore.ai/projects/graph-analyser-userguide/en/latest/user-guide.html#capturing-ipu-reports), set precision to [fp16/half](https://github.com/graphcore/tutorials/tree/master/tutorials/pytorch/mixed_precision), if needed run [pipeline-parallel](https://github.com/graphcore/tutorials/tree/master/tutorials/pytorch/pipelining)

This information is from https://github.com/graphcore/tutorials/blob/master/tutorials/pytorch/basics/walkthrough.ipynb.

## Config.py

```python
class ComputeMode(Enum):
    CPU   = 0
    #...
    IPU   = 5
```

## Imports

```python
import poptorch
```

## Loss

We will build a simple CNN model for a classification task. To do so, we can
simply use PyTorch's API, including `torch.nn.Module`. The difference from
what we're used to with pure PyTorch is the _**loss computation_, which has to
be part of the `forward` function.**

```python
    def forward(self, x, labels=None):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.norm(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        # The model is responsible for the calculation
        # of the loss when using an IPU. We do it this way:
        if self.training:
            return x, self.loss(x, labels)
        return x
```

## Data Loader

Use **poptorch.DataLoader**.

```python
opts = poptorch.Options()

train_dataloader = poptorch.DataLoader(
    opts, train_dataset, batch_size=16, shuffle=True, num_workers=20
)
```

> ***Note for IPU benchmarking***:
>
> The warmup time can be avoided by calling `training_model.compile(data,
> labels)` before any other call to the model. If not, the first call will
> include the compilation time, which can take few minutes.
>
> ```python
> # Warmup
> print("Compiling + Warmup ...")
> training_model.compile(data, labels)
> ```

See tutorials/pytorch/efficient_data_loading/walkthrough.ipynb for more infomation

## Optimizer

```python
optimizer = poptorch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

## TrainingModel

```python
poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optimizer)
```

## Training Loop

```python
epochs = 5
for epoch in tqdm(range(epochs), desc="epochs"):
    total_loss = 0.0
    for data, labels in tqdm(train_dataloader, desc="batches", leave=False):
        output, loss = poptorch_model(data, labels)
        total_loss += loss
```

The model is now trained! There's no need to retrieve the weights from the
device as you would by calling `model.cpu()` with PyTorch. PopTorch has
managed that step for us. We can now save and evaluate the model.
