# Waits for the overnight run to finish, then confirms the two promoted changes
# compose.
#
# Both gains were measured in isolation: null-move pruning was tested with the v1
# weights, and the v2 weights were tested with null-move off. Elo is usually
# roughly additive but interactions are real -- null-move prunes on the static
# evaluation, so a changed evaluation can change what it prunes. This measures
# the configuration the engine actually plays.

$log = "confirm.log"
function Announce($text) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $text
    Write-Output $line
    Add-Content -Path $log -Value $line
}

Announce "waiting for the overnight run (PID 19736) to release the machine"
try { Wait-Process -Id 19736 -ErrorAction Stop } catch { }
Announce "machine free, starting confirmation"

# New default (v2 weights + null move) against the old one (v1 weights, no null).
# Expect roughly +88 if the two gains simply add.
$r = .\target\release\gauntlet.exe --pairs 400 --movetime 1000 --a hand --b v1 --no-null-b --seed 611 2>$null |
     Select-String "vs hand-v1"
Announce "COMBINED $r"

# And against the spec weights, for the headline number.
$r = .\target\release\gauntlet.exe --pairs 300 --movetime 1000 --a hand --b spec --no-null-b --seed 612 2>$null |
     Select-String "vs hand-spec"
Announce "TOTAL-vs-SPEC $r"

Announce "=== confirmation complete ==="
