set ROOT_PATH=%CD%

cd %ROOT_PATH%\resource\program\
call COPY_SDK_INCLUDE_FILE.bat
cd %ROOT_PATH%

cd %ROOT_PATH%\thirdparty\
call INSTALL_BUILD_ALL.bat
cd %ROOT_PATH%

cd %ROOT_PATH%\sources\
call GenerateProject.bat
cd %ROOT_PATH%

pause