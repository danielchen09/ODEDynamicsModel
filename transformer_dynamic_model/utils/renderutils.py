import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def generate_video(frames, name='test.mp4'):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    # Switch to headless 'Agg' to inhibit figure rendering.
    matplotlib.use('Agg')
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


def render_env(env, steps, name='render.mp4'):
    frames = []
    action_spec = env.action_space
    random_state = np.random.RandomState(1337)
    for _ in range(steps):
        action = random_state.uniform(action_spec.low, action_spec.high, action_spec.shape)
        env.step(action)
        frames.append(env.render())
    generate_video(frames, name)