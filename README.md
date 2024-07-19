
# AI Framework

A simple AI learner framework based on Jeremy Howard's fastai course and pytorch.
### Installation

```bash
pip install 
```

# Learner operation

There is **Learner** that defines a *training loop* and its operations can be extended through the use of pre-existing **Callbacks** and **Hooks** or through the implementation of new ones.  
The learner can be used to train or eval a neural network model.


### Example usage
Basic usage: after defined a model and some useful callbacks, instantiate a Learner and execute it's **fit(epochs)** method.

```python

metrics = MetricsCB(accuracy=MulticlassAccuracy())
cbs = [get_device_cb(device='cpu'), get_metrics_cb(), get_progress_cb(plot=True)]
# cbs = [astats]

learn = Learner(model, dlsForConvolutional,
                F.cross_entropy, lr=0.01, callbacks=cbs)
learn.fit(3)
```
### Callbacks

There are several callbacks (and hooks), implemented and grouped into different categories:  

* **initialization** to transform data batches before training 
* **metrics** to measure accuracy, loss functions values during the training
* **plot** to draw plots of various statistics
* **device** used to select device (CPU, GPU, ..) where the training is performed
* **scheduler** to define, configure and execute schedulers
* **utilities** various features, ex.: memory clean callbacks





### Training a neural net

TODO::example..

### Running tests

To run the unit tests ...

```bash
python -m pytest
```

### License

MIT
