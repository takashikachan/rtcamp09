set GEMERATOR="Visual Studio 14 Win64"
set ARCHETECTURE=Win32
set OUTPUT_DIR_NAME=_builds
set GENERATOR_TOOLSET=host=x64
cmake -B %OUTPUT_DIR_NAME% -G %GEMERATOR% -T %GENERATOR_TOOLSET%
pause