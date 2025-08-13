#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Script PowerShell equivalente ao Makefile para Epocle Edge ML

.DESCRIPTION
    Fornece comandos para desenvolvimento, teste e empacotamento do projeto

.PARAMETER Target
    Comando a ser executado (help, install, test, lint, format, clean, data, demo, ci, package, release)

.EXAMPLE
    .\make.ps1 help
    .\make.ps1 test
    .\make.ps1 format
#>

param(
    [Parameter(Position=0)]
    [string]$Target = "help"
)

function Show-Help {
    Write-Host "Comandos disponíveis:" -ForegroundColor Green
    Write-Host "  help      - Mostrar esta ajuda" -ForegroundColor Yellow
    Write-Host "  install   - Instalar dependências" -ForegroundColor Yellow
    Write-Host "  test      - Executar testes com cobertura" -ForegroundColor Yellow
    Write-Host "  lint      - Verificar formatação" -ForegroundColor Yellow
    Write-Host "  format    - Formatar código com black e isort" -ForegroundColor Yellow
    Write-Host "  clean     - Limpar arquivos temporários" -ForegroundColor Yellow
    Write-Host "  data      - Gerar dados sintéticos" -ForegroundColor Yellow
    Write-Host "  demo      - Executar demo de treinamento online" -ForegroundColor Yellow
    Write-Host "  ci        - Executar CI completo (format, lint, test)" -ForegroundColor Yellow
    Write-Host "  package   - Criar pacote de distribuição" -ForegroundColor Yellow
    Write-Host "  release   - Preparar release (clean, test, package)" -ForegroundColor Yellow
}

function Install-Dependencies {
    Write-Host "Instalando dependências..." -ForegroundColor Green
    pip install -e .
    pip install -r requirements.txt
}

function Run-Tests {
    Write-Host "Executando testes com cobertura..." -ForegroundColor Green
    python -m pytest tests/ --cov=src --cov-report=term-missing -v
}

function Run-Lint {
    Write-Host "Verificando formatação..." -ForegroundColor Green
    python -m flake8 src/ tests/ examples/ --max-line-length=88 --ignore=E203,W503
}

function Format-Code {
    Write-Host "Formatando código..." -ForegroundColor Green
    python -m black src/ tests/ examples/ notebooks/
    python -m isort src/ tests/ examples/ notebooks/
}

function Clean-Project {
    Write-Host "Limpando arquivos temporários..." -ForegroundColor Green
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    if (Test-Path "*.egg-info") { Remove-Item -Recurse -Force "*.egg-info" }
    if (Test-Path "artifacts") { Get-ChildItem "artifacts" | Remove-Item -Force }
    if (Test-Path "data") { Get-ChildItem "data" | Remove-Item -Force }
    Get-ChildItem -Recurse -Include "*.pyc" | Remove-Item -Force
    Get-ChildItem -Recurse -Include "__pycache__" | Remove-Item -Recurse -Force
}

function Generate-Data {
    Write-Host "Gerando dados sintéticos..." -ForegroundColor Green
    python examples/synthetic_data.py --num-samples 1000 --num-features 20 --num-classes 3 --output data/synthetic_data.npz
}

function Run-Demo {
    Write-Host "Executando demo de treinamento online..." -ForegroundColor Green
    python examples/train_online.py --epochs 5 --batch-size 32 --use-dp --use-ewc
}

function Run-CI {
    Write-Host "Executando CI completo..." -ForegroundColor Green
    Format-Code
    Run-Lint
    Run-Tests
}

function Create-Package {
    Write-Host "Criando pacote de distribuição..." -ForegroundColor Green
    python -m build
}

function Prepare-Release {
    Write-Host "Preparando release..." -ForegroundColor Green
    Clean-Project
    Run-Tests
    Create-Package
}

# Main execution
switch ($Target.ToLower()) {
    "help" { Show-Help }
    "install" { Install-Dependencies }
    "test" { Run-Tests }
    "lint" { Run-Lint }
    "format" { Format-Code }
    "clean" { Clean-Project }
    "data" { Generate-Data }
    "demo" { Run-Demo }
    "ci" { Run-CI }
    "package" { Create-Package }
    "release" { Prepare-Release }
    default {
        Write-Host "Comando desconhecido: $Target" -ForegroundColor Red
        Show-Help
        exit 1
    }
}
