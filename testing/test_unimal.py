import os
import glob
from dm_control import mujoco
from dm_control.rl.control import Environment
from dm_control.suite.base import Task
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def generate_video(frames, name='test.mp4'):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=1000 / 30, blit=True, repeat=False)
    anim.save(name)

data_dir = '../unimals_100/train'
xml_raw_paths = glob.glob(data_dir + '/xml/*.xml')
env_names = [xml_raw_path.split('/')[-1].split('.')[0] for xml_raw_path in xml_raw_paths]

env_name = env_names[0]
env_xml = env_name + '.xml'
print(env_name)
physics = mujoco.Physics.from_xml_path('../unimals_100/train/xml/floor-1409-0-3-01-15-56-55.xml')

class TestTask(Task):
    def __init__(self):
        super().__init__()
    
    def get_observation(self, physics):
        return [0, 0]

    def get_reward(self, physics):
        return 0

env = Environment(physics, TestTask())
steps = 100
frames = []
action_spec = env.action_spec()
random_state = np.random.RandomState(1337)
cam = mujoco.MovableCamera(physics)

def get_names(physics, type):
    name_adrs = getattr(physics.model, f'name_{type}adr')
    names = physics.model.names.decode('utf-8')
    return [names[i:names.index('\x00', i)] for i in name_adrs]
    
for _ in range(steps):
    action = random_state.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
    env.step(action)
    frames.append(physics.render(camera_id=0))

generate_video(frames)
