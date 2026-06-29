# MinGW-w64 Installation Script for Rust
# This installs the minimal toolchain needed for Rust compilation

Write-Host "Installing MinGW-w64 via winget..." -ForegroundColor Green

# Try winget first (Windows 11/10 with App Installer)
try {
    winget install --id=MSYS2.MSYS2 -e --accept-source-agreements --accept-package-agreements
    
    Write-Host "`nMSYS2 installed. Now installing MinGW toolchain..." -ForegroundColor Green
    
    # Install MinGW toolchain via MSYS2
    & "C:\msys64\usr\bin\bash.exe" -lc "pacman -Sy --noconfirm mingw-w64-x86_64-toolchain"
    
    # Add to PATH
    $mingwPath = "C:\msys64\mingw64\bin"
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    
    if ($currentPath -notlike "*$mingwPath*") {
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$mingwPath", "User")
        Write-Host "`nAdded MinGW to PATH: $mingwPath" -ForegroundColor Green
    }
    
    Write-Host "`n✓ MinGW installation complete!" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "1. Restart your terminal/IDE" -ForegroundColor Yellow
    Write-Host "2. Verify with: gcc --version" -ForegroundColor Yellow
    Write-Host "3. Then run: cargo build --release" -ForegroundColor Yellow
    
} catch {
    Write-Host "`nwinget not available. Please install manually:" -ForegroundColor Red
    Write-Host "1. Download MSYS2: https://www.msys2.org/" -ForegroundColor Yellow
    Write-Host "2. Run installer (msys2-x86_64-*.exe)" -ForegroundColor Yellow
    Write-Host "3. Open MSYS2 terminal and run: pacman -Sy mingw-w64-x86_64-toolchain" -ForegroundColor Yellow
    Write-Host "4. Add C:\msys64\mingw64\bin to your PATH" -ForegroundColor Yellow
}
