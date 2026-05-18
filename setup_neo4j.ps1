# Neo4j 5.20 Community Standalone Setup for Windows
# Run this script after downloading Neo4j from: https://neo4j.com/download-center/community/

$NEO4J_HOME = "C:\neo4j-5.20.0"
$NEO4J_DOWNLOAD_URL = "https://neo4j.com/download-center/community/files/neo4j-community-5.20.0-windows.zip"
$NEO4J_ZIP = "$env:TEMP\neo4j-5.20.0-windows.zip"

Write-Host "Neo4j 5.20 Community Standalone Setup" -ForegroundColor Cyan
Write-Host "======================================`n"

# Step 1: Download (if not already present)
if (-not (Test-Path $NEO4J_ZIP)) {
    Write-Host "Downloading Neo4j 5.20..." -ForegroundColor Yellow
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $NEO4J_DOWNLOAD_URL -OutFile $NEO4J_ZIP -ErrorAction Stop
    Write-Host "Download complete." -ForegroundColor Green
} else {
    Write-Host "Neo4j zip already found at $NEO4J_ZIP" -ForegroundColor Green
}

# Step 2: Extract
if (-not (Test-Path $NEO4J_HOME)) {
    Write-Host "`nExtracting to $NEO4J_HOME..." -ForegroundColor Yellow
    Expand-Archive -Path $NEO4J_ZIP -DestinationPath "C:\" -Force
    Write-Host "Extraction complete." -ForegroundColor Green
} else {
    Write-Host "Neo4j already extracted at $NEO4J_HOME" -ForegroundColor Green
}

# Step 3: Configure neo4j.conf
$CONF_FILE = "$NEO4J_HOME\conf\neo4j.conf"
Write-Host "`nConfiguring $CONF_FILE..." -ForegroundColor Yellow

$CONFIG = @"
# Memory settings (free tier: 512m initial, 2GB max)
dbms.memory.heap.initial_size=512m
dbms.memory.heap.max_size=2G

# Network
server.default_listen_address=0.0.0.0
server.default_advertised_address=localhost

# Plugins (APOC)
server.plugins.directory=plugins
dbms.security.procedures.unrestricted=apoc.*

# Database name
initial.dbms.default_database=neo4j

# Auth
server.auth.authentication_providers=basic

# Logging
server.directories.logs=logs
"@

Add-Content -Path $CONF_FILE -Value "`n$CONFIG" -ErrorAction SilentlyContinue
Write-Host "Configuration updated." -ForegroundColor Green

# Step 4: Download APOC plugin
Write-Host "`nDownloading APOC plugin..." -ForegroundColor Yellow
$APOC_JAR = "C:\neo4j-5.20.0\plugins\apoc-5.20.0-core.jar"
$PLUGINS_DIR = "C:\neo4j-5.20.0\plugins"

if (-not (Test-Path $PLUGINS_DIR)) {
    New-Item -ItemType Directory -Path $PLUGINS_DIR -Force | Out-Null
}

if (-not (Test-Path $APOC_JAR)) {
    try {
        $APOC_URL = "https://github.com/neo4j/apoc/releases/download/5.20.0/apoc-5.20.0-core.jar"
        Invoke-WebRequest -Uri $APOC_URL -OutFile $APOC_JAR -ErrorAction Stop
        Write-Host "APOC plugin downloaded." -ForegroundColor Green
    } catch {
        Write-Host "Note: APOC plugin download failed (optional for basic use)" -ForegroundColor Yellow
    }
} else {
    Write-Host "APOC plugin already present." -ForegroundColor Green
}

# Step 5: Set auth
Write-Host "`nSetting authentication (neo4j / cmg-demo-pass)..." -ForegroundColor Yellow
& "$NEO4J_HOME\bin\neo4j-admin.bat" dbms set-initial-password cmg-demo-pass 2>&1 | Out-Null
Write-Host "Initial password set." -ForegroundColor Green

# Step 6: Show next steps
Write-Host "`n"
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Start Neo4j:" -ForegroundColor Yellow
Write-Host "   & 'C:\neo4j-5.20.0\bin\neo4j.bat' console"
Write-Host ""
Write-Host "2. Verify it's running:" -ForegroundColor Yellow
Write-Host "   http://localhost:7474 (neo4j / cmg-demo-pass)"
Write-Host "   bolt://localhost:7687"
Write-Host ""
Write-Host "3. Load schema constraints:" -ForegroundColor Yellow
Write-Host "   python -m etl.build_graph --load-schema-only"
Write-Host ""
Write-Host "Neo4j Home: $NEO4J_HOME" -ForegroundColor Cyan
