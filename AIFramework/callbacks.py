class Callback():
    order = 0


def run_callbacks(callbacks, method_name, learn=None):
    for callback in sorted(callbacks, key=attrgetter('order')):
        method = getattr(callback, method_name, None)
        if method is not None:
            method(learn)
