from yacs.config import CfgNode as CN

_C = CN()

_C.env = CN()
_C.env.SELECT_OBS = [
    'geom_xpos',
    'geom_xvel',
    'geom_xquat',
    'body_xpos',
    'body_xquat',
    'body_xvel',
    'geom_type',
    'geom_pos',
    'geom_quat',
    'geom_size',
    'geom_friction',
    'body_pos',
    'body_ipos',
    'body_iquat',
    'body_mass',
    'jnt_qpos',
    'jnt_qvel',
    'jnt_pos',
    'jnt_range',
    'jnt_axis',
    'jnt_gear',
    'jnt_armature',
    'jnt_damping'
]

_C.model = CN()
_C.model.MAX_SEQ_LEN = 15
_C.model.DROPOUT = 0.1
_C.model.GLOBAL_EMB_DIM = 128
_C.model.encoder = CN()
_C.model.encoder.DIM = 128
_C.model.encoder.N_LAYERS = 5
_C.model.encoder.MLP_DIM = 1024
_C.model.encoder.N_HEADS = 2
_C.model.decoder = CN()
_C.model.decoder.HIDDEN_DIMS = [64, 64]

_C.buffer = CN()
_C.buffer.CAPACITY = 10000
_C.buffer.ALPHA = 0.1
_C.buffer.BETA = 1.0

config = _C