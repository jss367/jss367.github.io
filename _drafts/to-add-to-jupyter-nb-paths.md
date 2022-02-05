


%%javascript
var kernel = Jupyter.notebook.kernel
kernel.execute('kernel_name = ' + '"' + kernel.name + '"')


print(kernel_name)


from jupyter_client import kernelspec
spec = kernelspec.get_kernel_spec(kernel_name)
print(spec.resource_dir)
# /path/to/my/kernel



https://stackoverflow.com/questions/43759543/how-to-get-active-kernel-name-in-jupyter-notebook












You could be tripped up by having kernels in different locations:

~/.local/share/jupyter/kernels/tf

~/miniconda3/envs/tf/share/jupyter/kernels/python3

These are both "tf", but they are different.





python -m ipykernel install --user --name tf2 --display-name "TF2"






