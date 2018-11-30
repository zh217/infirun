import abc
import importlib


class Schematic(abc.ABC):
    @abc.abstractmethod
    def _get_schema(self):
        pass

    def get_schema(self):
        schema = self._get_schema()
        schema['type'] = [type(self).__module__, type(self).__name__]
        return schema

    @staticmethod
    @abc.abstractmethod
    def from_schema(schema):
        pass


def from_schema(schema):
    mod_name, cls_name = schema['type']
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls.from_schema(schema)
