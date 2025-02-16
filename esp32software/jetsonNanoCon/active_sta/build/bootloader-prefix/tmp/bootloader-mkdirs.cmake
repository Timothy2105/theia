# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/river/esp/esp-idf-v4.3/components/bootloader/subproject")
  file(MAKE_DIRECTORY "/home/river/esp/esp-idf-v4.3/components/bootloader/subproject")
endif()
file(MAKE_DIRECTORY
  "/home/river/Documents/GitHub/theia/esp32software/jetsonNanoCon/active_sta/build/bootloader"
  "/home/river/Documents/GitHub/theia/esp32software/jetsonNanoCon/active_sta/build/bootloader-prefix"
  "/home/river/Documents/GitHub/theia/esp32software/jetsonNanoCon/active_sta/build/bootloader-prefix/tmp"
  "/home/river/Documents/GitHub/theia/esp32software/jetsonNanoCon/active_sta/build/bootloader-prefix/src/bootloader-stamp"
  "/home/river/Documents/GitHub/theia/esp32software/jetsonNanoCon/active_sta/build/bootloader-prefix/src"
  "/home/river/Documents/GitHub/theia/esp32software/jetsonNanoCon/active_sta/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/river/Documents/GitHub/theia/esp32software/jetsonNanoCon/active_sta/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/river/Documents/GitHub/theia/esp32software/jetsonNanoCon/active_sta/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
