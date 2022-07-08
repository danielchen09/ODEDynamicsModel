from dm_control import mujoco
import numpy as np

physics = mujoco.Physics.from_xml_path('../unimals_100/train/xml/floor-1409-0-3-01-15-56-55.xml')
q = np.zeros((4, 1))
mujoco.mju_mat2Quat(q, physics.data.geom_xmat[0])
breakpoint()