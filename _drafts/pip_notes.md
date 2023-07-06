You should install local packages with `pip install -e .` or `pip install -qe .`

You need a setup.py to do it

If you depend on a local package and you don't want to have to reinstall it when you change the code.


Also allow acceptance tests to see the python project that is being tested

Every change you make to the library is picked up where you're using it.





pip install specifically for Python, brew installations are system-wide.

In the case of protobuf:

The pip installation provides the Python package google.protobuf that includes Python bindings for Protocol Buffers and can be imported in your Python code using import google.protobuf.

The brew installation provides the Protocol Buffers compiler (protoc), which is a command-line tool for generating bindings for multiple languages from .proto files. It also provides the protobuf C++ libraries, which might be needed for certain system-level applications or for developing in languages like C++.

If you're just using protobuf in Python, you should be able to just `pip install protobuf`. Also, if you use TensorFlow, you'll have to have the pip version.
