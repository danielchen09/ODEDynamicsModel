from datetime import datetime
from tensorboard import program
import tensorflow as tf

def get_date():
    return datetime.now().strftime("%m%d%H%M%S")

class Plotter:
    def __init__(self, port=6006, log_name=None, steps=0, model=None):
        self.name = get_date() if log_name is None else log_name.split('.')[0]
        self.log_dir = './logs/' + self.name
        self.writer = tf.summary.create_file_writer(self.log_dir, flush_millis=1000)
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir, '--reload_interval', '1', '--port', f'{port}'])
        self.url = tb.launch()
        print(f"Tensorflow listening on {self.url} log_dir: {self.log_dir}")
        self.steps = steps
        self.epochs = 0

    def log_loss(self, name, value):
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=self.steps)
            
    def step(self):
        self.steps += 1
    
    def step_epoch(self):
        self.epochs += 1