from dm_control import mujoco
import numpy as np

def get_names(physics, obj_type, filter_fn=None):
    if filter_fn is None:
        filter_fn = lambda x: True
    name_adrs = getattr(physics.model, f'name_{obj_type}adr')
    all_names = physics.model.names.decode('utf-8')
    names = []
    for i in name_adrs:
        name = all_names[i:all_names.index('\x00', i)]
        if filter_fn(name):
            names.append(name)
    return names

def name2id(physics, obj_type, name):
    obj_type = getattr(mujoco.mjtObj, f'mjOBJ_{obj_type.upper()}')
    return mujoco.mj_name2id(physics.model._model, obj_type, name)

def get_idxs(physics, obj_type):
    def filter_fn(name):
        return any(substr in name for substr in ['torso', 'limb'])
    names = get_names(physics, obj_type, filter_fn=filter_fn)
    return [name2id(physics, obj_type, name) for name in names]

def get_vel(physics, idxs, type):
    type = getattr(mujoco.mjtObj, f'mjOBJ_{type.upper()}')
    vels = []
    for idx in idxs:
        buf = np.zeros((6, 1))
        mujoco.mj_objectVelocity(physics.model._model, physics.data._data, type, idx, buf, 0)
        vels.append(buf.reshape(-1))
    return np.vstack(vels)

def mat2quat(mat):
    quats = []
    for row in mat:
        buf = np.zeros((4, 1))
        mujoco.mju_mat2Quat(buf, row)
        quats.append(buf.reshape(-1))
    return np.vstack(quats)

def get_action_bound(physics):
    ctrl_range = physics.model.actuator_ctrlrange.copy().astype(np.float32)
    return ctrl_range.T