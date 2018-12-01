def pipe_step(wrapped):
    class Whatever:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __call__(self, *args, **kwargs):
            pass

        def serialize(self):
            pass

    return Whatever()
