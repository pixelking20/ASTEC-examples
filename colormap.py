from matplotlib import pyplot as plt
class ColorMap(object):
    def __init__(self, values, cmap='jet', default='w'):
        values = sorted(values)
        if None in values: values.remove(None)        
        if all([isinstance(i, int) for i in values]):
            self.values = range(values[0], values[-1]+1)
            self.cmap = plt.cm.get_cmap(cmap, len(self.values))
        elif all([isinstance(i, float) for i in values]):
            self.values = None   
            self.cmap = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=values[0], vmax=values[-1]), cmap=cmap)
        else:
            self.values = values
            self.cmap = plt.cm.get_cmap(cmap, len(self.values))
        self.default = default
        
    def __call__(self, value):
        if value is None:
            return self.default
        elif self.values is None:
            return self.cmap.cmap(self.cmap.norm(value))
        else:
            if not value in self.values:
                raise ValueError('`value` parameter')
            return self.cmap(self.values.index(value))