# Bootstrap entry point - delegates to scripts/ci/build.sh
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$ScriptDir\scripts\ci\build.sh" @args
