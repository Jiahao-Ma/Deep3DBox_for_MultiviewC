import json
import numpy as np
"""
Enables writing json with numpy array to files
"""
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

"""
Class hold the average dimension for a class, regress value is the residual
"""
class ClassAverage(object):
    def __init__(self, classes=['Cow']) -> None:
        super().__init__()
        self.save_class_average_path = r'./data/classes_averages.json'
        self.dimension_map = {}
        for cls in classes:
            cls_ = cls.lower()
            if cls_ in self.dimension_map.keys():
                continue
            self.dimension_map[cls_] = {}
            self.dimension_map[cls_]['count'] = 0
            self.dimension_map[cls_]['total'] = np.zeros(3, dtype=np.float32)

    def dump_to_file(self):
        with open(self.save_class_average_path, 'w') as f:
            json.dump(self.dimension_map, f,  cls=NumpyEncoder, indent=4)

    
    def add_item(self, cls_, dimension):
        cls_ = cls_.lower()
        self.dimension_map[cls_]['count'] += 1
        self.dimension_map[cls_]['total'] += dimension
    
    def get_item(self, cls_):
        cls_ = cls_.lower()
        return self.dimension_map[cls_]['total'] / self.dimension_map[cls_]['count']