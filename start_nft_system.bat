@echo off

echo.
start "Blockchain Node" cmd /k "npx hardhat node"
timeout /t 5 /nobreak >nul
start "AI Oracle API" cmd /k "cd backend && py -3.12 api.py"
start "Next.js Frontend" cmd /k "cd frontend && npm run dev"

echo.

pause
