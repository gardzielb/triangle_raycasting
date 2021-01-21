# Compilation

Project can be compiled with Cmake version >= 3.17.

# Launching

Project accepts up to 4 command line arguments:
* device: _cpu_ or _gpu_
* model name: a name of .obj file to load, without extension, i.e. _exampleModel_. A corresponding .obj file must be placed inside _model_ directory.
* model color: in RGB format, 3 floats of range [0.0, 1.0] separated with spaces
* model shininess: float value

The first two are required, the other two have default values.