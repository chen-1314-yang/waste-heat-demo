@echo off
setlocal
cd /d "%~dp0"

rem 让捆绑版 git 能找到 https 传输组件（2026-08-14 修复）
set "GIT_EXEC_PATH=C:\Users\23549\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\mingw64\bin"

set "REPO=%~dp0"
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"
git config --global --add safe.directory "%REPO%" >nul 2>&1

git config user.name "chen-1314-yang"
git config user.email "chen-1314-yang@users.noreply.github.com"

if not exist .git (
  echo Initializing git repository ...
  git init -b main
)

git add .
git commit -m "init: waste heat intelligent decision demo"
git branch -M main

echo.
echo Adding remote and pushing to GitHub ...
echo If a login window opens, sign in with your GitHub account.
git remote remove origin 2>nul
git remote add origin https://github.com/chen-1314-yang/waste-heat-demo.git
git push -u origin main

if %errorlevel%==0 (
  echo.
  echo PUSH OK - next step: open https://share.streamlit.io and deploy.
) else (
  echo.
  echo PUSH FAILED - check the message above.
)
pause
