import json
import numpy as np
from munch import DefaultMunch

def data_type_convert(str_or_dt):
    if isinstance(str_or_dt, str):
        if len(str_or_dt) > 1:
            return str_or_dt
        assert 0, f'invalid str type: {type(str_or_dt)}!'
        
    elif isinstance(str_or_dt, np.dtype):
        [(_,dt)] = str_or_dt.descr
        # print(f'date type: {dt}')
        return dt
    else:
        assert 0,f'unrecognized type: {str_or_dt}'

class BackendImage(object):
    def __init__(self, data, meta=None):
        self.init_meta(meta)
        self.init_data(data)

    def init_meta(self,meta):
        if isinstance(meta, DefaultMunch):
            self.meta_dict = meta
        elif isinstance(meta, dict):
            self.meta_dict = DefaultMunch.fromDict(meta)
        elif isinstance(meta, bytes):
            self.meta_dict = DefaultMunch.fromDict(json.loads(meta))
        elif meta is None: 
            print('meta auto init')
        else:
            assert f'error meta data type: {type(meta)}!'
            
    def init_data(self,data):
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, bytes):
            data_type = self.meta_dict.DataType
            dtype=data_type_convert(data_type)
            shape_x, shape_y, shape_z = self.meta_dict.Shape.X, self.meta_dict.Shape.Y, self.meta_dict.Shape.Z
            self.data = np.frombuffer(data, dtype).reshape(shape_z,shape_y, shape_x).astype(np.float32)
        elif data == None:
            assert 'data is None!'
        else:
            assert f'error data type: {type(data)}!'