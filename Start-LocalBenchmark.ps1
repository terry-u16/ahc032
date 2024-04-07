Write-Host "[Compile]"
cargo build --release
Move-Item ../target/release/ahc032.exe . -Force
Write-Host "[Run]"
$env:DURATION_MUL = "1.5"
dotnet marathon run-local
#./relative_score.exe -d ./data/results -o min