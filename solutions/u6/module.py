from abc import ABC, abstractmethod


class Module(ABC):
    """
    Abstract neural network module.  Not to mix with a Python module.

    We use the Module class to encapsulate the forward calculation of
    a network component together with the corresponding parameters.
    """

    # The @abstractmethod annotation is used to state that the forward
    # method should be implemented in the subclass.
    @abstractmethod
    def forward(self):
        """Defines the forward computation."""
        raise NotImplementedError

    def register(self, attr_name: str, param):
        """Register the given tensor parameter or sub-module with
        its own parameters.  In case the input argument is a tensor
        with the requires_grad attribute, it is set to True.

        Args:
            attr: the attribute name to store the given parameter at
            param: either a tensor parameter or a submodule
                (Union[TT, Module])
        """
        # If attr_name = "attr", then the following line is equivalent to
        #   self.attr = param
        setattr(self, attr_name, param)
        # We set the `requires_grad` flag to True in case of a tensor
        if hasattr(param, "requires_grad"):
            param.requires_grad = True
        try:
            self.param_list.append(param)
        except AttributeError:
            self.param_list = [param]

    def params(self):
        """Return the list of parameters (tensors) used in the model."""
        # The outer try is used because a module can be defined with
        # no parameters, in which case self.param_list raises the
        # AttributeError exception.
        try:
            result = []
            for param in self.param_list:
                try:
                    result += param.params()
                except AttributeError:
                    result.append(param)
            return result
        except AttributeError:
            return []
