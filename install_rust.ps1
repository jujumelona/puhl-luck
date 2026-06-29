# Rust Installation Script for Windows
# Run this in PowerShell as Administrator

Write-Host "Installing Rust for Windows..." -ForegroundColor Green

# Download rustup installer
$rustupUrl = "https://win.rustup.rs/x86_64"
$rustupInstaller = "$env:TEMP\rustup-init.exe"

Write-Host "Downloading rustup installer..." -ForegroundColor Yellow
Invoke-WebRequest -Uri $rustupUrl -OutFile $rustupInstaller

# Run installer
Write-Host "Running rustup installer..." -ForegroundColor Yellow
Write-Host "Please follow the prompts and accept defaults (option 1)" -ForegroundColor Cyan
& $rustupInstaller

# Add to PATH for current session
$cargoPath = "$env:USERPROFILE\.cargo\bin"
if (Test-Path $cargoPath) {
    $env:Path = "$cargoPath;$env:Path"
    Write-Host "Rust installed successfully!" -ForegroundColor Green
    Write-Host "Cargo version:" -ForegroundColor Cyan
    cargo --version
    rustc --version
} else {
    Write-Host "Installation may have failed. Please restart PowerShell and try: cargo --version" -ForegroundColor Red
}

# Install maturin
Write-Host "`nInstalling maturin (Python-Rust build tool)..." -ForegroundColor Yellow
pip install maturin

Write-Host "`nSetup complete! Please restart your terminal/IDE for PATH changes to take effect." -ForegroundColor Green
Write-Host "Then run: cargo --version" -ForegroundColor Cyan
