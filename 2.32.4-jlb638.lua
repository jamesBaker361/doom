help(
[[
Simple DirectMedia Layer is a cross-platform development
 library designed to provide low level access to audio,
  keyboard, mouse, joystick, and graphics hardware via OpenGL and Direct3D.
]])

whatis("a cross-platform library designed to make it easy to write multi-media software, such as games and emulators")



local base = pathJoin("/projects/community/sdl/2.32.4/jlb638/")
prepend_path("PATH", pathJoin(base, "bin"))
prepend_path("LIBRARY_PATH", pathJoin(base, "lib"))
prepend_path("CPATH", pathJoin(base, "include"))
prepend_path("CPLUS_INCLUDE_PATH", pathJoin(base, "include"))
prepend_path("LD_LIBRARY_PATH", pathJoin(base, "lib"))

load("gcc/5.4")
