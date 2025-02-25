import torch

# Register a custom library and print operator
from torch.library import Library, impl
lib = Library("mylib", "DEF")
lib.define("print(str message) -> ()")
@impl(lib, "print", "CompositeExplicitAutograd")
def print_op(message):
    print(message)
    return

t1 = torch.randn(10, 10)
t2 = torch.randn(10, 10)

def foo(x, y):
    a = torch.sin(x)
    torch.ops.mylib.print(f"Hello from torch.compile: {a.shape}")
    b = torch.cos(y)
    return a + b

opt_foo = torch.compile(foo, fullgraph=True)
print(opt_foo(t1, t2))
