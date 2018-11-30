import abc


class Schematic(abc.ABC):
    @abc.abstractmethod
    def get_schema(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def from_schema(schema):
        """
        idempotency: Schematic.from_schema(a.get_schema()) == a.get_schema()
        :param schema:
        :return:
        """
        pass

    def restore(self):
        pass
