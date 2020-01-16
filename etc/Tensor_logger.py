try:
    import tensorflow as tf
    TENSORBOARD = True
except ImportError:
    print('no tensorflow found. set use_tensorboard = False')
    TENSORBOARD = False

try:
    import visdom
    VISDOM = True
except ImportError:
    print('no visdom found. set visdom_port = None')
    VISDOM = False

try:
    import nsml
except ImportError:
    pass


class Logger:
    def __init__(self, visdom_port=None, log_dir=None, use_nsml=False):
        self.use_nsml = use_nsml
        if use_nsml:
            self.vis = nsml.Visdom(visdom=visdom)
            self.last = None #python -m visdom.server
        elif VISDOM and visdom_port:
            self.vis = visdom.Visdom(port=visdom_port)
            if not self.vis.check_connection():
                print('No visdom server found on port {}. set visdom_port = None'.format(visdom_port))
                self.vis = None
        else:
            self.vis = None
        self.use_visdom = use_nsml or visdom_port
        self.use_tensorboard = True if TENSORBOARD and log_dir is not None else False

        if not use_nsml and self.use_tensorboard:
            self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        if self.use_nsml:
            if self.last and self.last['step'] != step:
                nsml.report(**self.last, scope=None)
                self.last = None
            if self.last is None:
                self.last = {'step': step, 'iter': step, 'epoch': 1}
            self.last[tag] = value
        elif self.use_tensorboard:
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, step)

    def image_summary(self, data, opts):
        if self.use_visdom:
            self.vis.images(data, opts=opts,)
