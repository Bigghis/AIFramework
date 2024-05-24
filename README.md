
# AI Framework

A simple AI learner framework based on Jeremy Howard's fastai course ...

### Installation

```bash
pip install 
```

### Example usage

Basic example

```python

metrics = MetricsCB(accuracy=MulticlassAccuracy())

def_device = 'cuda' if torch.cuda.is_available() else 'cpu'

#print("DEVICE =", def_device)
cbs = [get_device_cb(device='cpu'), get_metrics_cb(), get_progress_cb(plot=True)]
# cbs = [astats]

learn = Learner(cnn2, dlsForConvolutional,
                F.cross_entropy, lr=0.01, callbacks=cbs)
```

### Training a neural net

TODO::example..

### Running tests

To run the unit tests ...

```bash
python -m pytest
```

### License

MIT
