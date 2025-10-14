@echo off
:: ===============================
:: Git push automation script (Windows .bat)
:: Equivalent of your bash script
:: ===============================

set /p commit_message=Pushing to git... Enter commit message: 

:: Check if input is '0'
if "%commit_message%"=="0" (
    echo Commit aborted.
    exit /b 1
)

:: Check if empty
if "%commit_message%"=="" (
    echo Commit message cannot be empty. Please provide a message.
    exit /b 1
)

echo.
echo Adding all changes...
git add .

echo.
echo Committing changes...
git commit -m "%commit_message%"

echo.
echo Pushing to 'main' branch...
git push origin main

echo.
echo Done! Changes pushed successfully.
pause
