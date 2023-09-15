is_simple_core=False

if is_simple_core:
    from mamingDL.core_simple import Variable
    from mamingDL.core_simple import Function
    from mamingDL.core_simple import using_config
    from mamingDL.core_simple import no_grad
    from mamingDL.core_simple import as_array
    from mamingDL.core_simple import as_variable
    from mamingDL.core_simple import setup_variable
else:
    from mamingDL.core import Variable
    from mamingDL.core import Function
    from mamingDL.core import using_config
    from mamingDL.core import no_grad
    from mamingDL.core import as_array
    from mamingDL.core import as_variable
    from mamingDL.core import setup_variable
    from mamingDL.core import Parameter
    from mamingDL.layers import Layer
    from mamingDL.models import Model
    from mamingDL.dataloaders import DataLoader

setup_variable()