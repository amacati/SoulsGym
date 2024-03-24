# Game Window build instructions

The game window module is not recompiled on installation since it requires Visual Studio and has several additional C++ dependencies. If you do want to rebuild it, please go through the following steps.

1. Install Visual Studio 2022
2. Open the `window_capture.sln` solution from this folder
3. Install `pybind11` in your current Python environment.
4. Set the include and library paths in the project settings to point to your current installation
5. Rebuild the whole solution. The `window_capture.pyd` extension is automatically copied to the _C directory in the last step of the build process