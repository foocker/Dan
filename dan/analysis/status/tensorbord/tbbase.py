from abc import abstractmethod, ABCMeta
  

class Base(metaclass=ABCMeta):
    """A series of interface."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def status_plot(self, *args, **kwargs):
        pass
