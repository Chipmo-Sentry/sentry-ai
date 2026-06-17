; Inno Setup script for the Sentry AI server.
; Build:  iscc /DAppVersion=0.1.0 installer\sentry-ai-server.iss
; (CI passes the version from the git tag.)
;
; Produces dist\ChipmoSentryAi-Setup.exe - an installer that:
;   - bundles the sentry-ai source + MediaMTX + cloudflared + nssm
;   - asks for the Railway backend URL, service token, Ollama URL, tunnel name
;   - on finish, runs setup-server.ps1: uv sync (downloads torch), writes .env,
;     installs the ingest + ai (+ tunnel) Windows services
;   - Ollama is NOT installed (use the user's existing install; we point at it)

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define AppName "Sentry AI Server"
#define AppPublisher "Chipmo"
#define AppId "{{A1C7E0D2-AI55-4B10-9E33-CHIPMOSENTRYAI}"

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\Sentry AI
DefaultGroupName=Sentry AI
DisableProgramGroupPage=yes
DisableDirPage=no
OutputDir=..\dist
OutputBaseFilename=ChipmoSentryAi-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; Services require admin.
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; sentry-ai source (the uv project; .venv is created at install by uv sync).
Source: "..\src\*";          DestDir: "{app}\app\src";     Flags: recursesubdirs ignoreversion
Source: "..\prompts\*";      DestDir: "{app}\app\prompts"; Flags: recursesubdirs ignoreversion
Source: "..\pyproject.toml"; DestDir: "{app}\app";         Flags: ignoreversion
Source: "..\uv.lock";        DestDir: "{app}\app";         Flags: ignoreversion
Source: "..\README.md";      DestDir: "{app}\app";         Flags: ignoreversion
; Management + setup scripts.
Source: "scripts\*";         DestDir: "{app}\scripts";     Flags: ignoreversion
; Config templates.
Source: "config\*";          DestDir: "{app}\config";      Flags: ignoreversion onlyifdoesntexist
; Bundled binaries (CI downloads these into installer\bin\ before compiling).
Source: "bin\mediamtx.exe";    DestDir: "{app}\bin"; Flags: ignoreversion
Source: "config\mediamtx.yml"; DestDir: "{app}\bin"; Flags: ignoreversion onlyifdoesntexist
Source: "bin\cloudflared.exe"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "bin\nssm.exe";        DestDir: "{app}\bin"; Flags: ignoreversion

[Icons]
Name: "{group}\Sentry AI - Status"; Filename: "powershell.exe"; \
  Parameters: "-NoExit -ExecutionPolicy Bypass -File ""{app}\scripts\server-control.ps1"" status"
Name: "{group}\Pairing - status"; Filename: "powershell.exe"; \
  Parameters: "-NoExit -ExecutionPolicy Bypass -File ""{app}\scripts\pairing.ps1"" status"
Name: "{group}\Pairing - re-pair (new code)"; Filename: "powershell.exe"; \
  Parameters: "-NoExit -ExecutionPolicy Bypass -File ""{app}\scripts\pairing.ps1"" pair"
Name: "{group}\Pairing - unpair"; Filename: "powershell.exe"; \
  Parameters: "-NoExit -ExecutionPolicy Bypass -File ""{app}\scripts\pairing.ps1"" unpair"
Name: "{group}\Re-run setup"; Filename: "powershell.exe"; \
  Parameters: "-NoExit -ExecutionPolicy Bypass -File ""{app}\scripts\setup-server.ps1"""
Name: "{group}\Check for updates"; Filename: "powershell.exe"; \
  Parameters: "-NoExit -ExecutionPolicy Bypass -File ""{app}\scripts\update.ps1"""
Name: "{group}\Uninstall"; Filename: "{uninstallexe}"

[Run]
; First-time setup: uv sync (downloads torch, 10-20 min) + write .env + install
; services. Shown on the Finish page, checked by default. Visible console.
Filename: "powershell.exe"; \
  Parameters: "-NoExit -ExecutionPolicy Bypass -File ""{app}\scripts\setup-server.ps1"" {code:GetSetupArgs}"; \
  Description: "Run first-time setup (downloads dependencies, installs services)"; \
  Flags: postinstall skipifsilent

[UninstallRun]
; Stop + remove services before files are deleted.
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\uninstall-services.ps1"""; \
  Flags: runhidden; RunOnceId: "RemoveSentryAiServices"

[Code]
var
  CfgPage: TInputQueryWizardPage;

procedure InitializeWizard;
begin
  CfgPage := CreateInputQueryPage(wpSelectDir,
    'Server configuration',
    'How this AI server connects to the Railway backend',
    'Generate a pairing code in superadmin (AI сервер -> Холболтын код). On an update you can leave the code blank to keep the existing pairing.');
  CfgPage.Add('Railway backend URL (e.g. https://sentry-backend-xxxx.up.railway.app):', False);
  CfgPage.Add('Pairing code (6 digits from superadmin):', False);
  CfgPage.Add('Ollama URL:', False);
  CfgPage.Add('cloudflared tunnel name (optional - leave blank to skip):', False);
  CfgPage.Values[2] := 'http://localhost:11434';
  CfgPage.Values[3] := '';
end;

procedure CurPageChanged(CurPageID: Integer);
var
  envLines: TArrayOfString;
  i, p: Integer;
  k, v, envPath, nodeId: string;
begin
  // On an update (re-install into the same folder), pre-fill the wizard from
  // the existing runtime config so the user doesn't re-type everything.
  if CurPageID = CfgPage.ID then
  begin
    nodeId := '';
    envPath := ExpandConstant('{app}\app\.env');
    if FileExists(envPath) then
    begin
      if LoadStringsFromFile(envPath, envLines) then
      begin
        for i := 0 to GetArrayLength(envLines) - 1 do
        begin
          p := Pos('=', envLines[i]);
          if p > 0 then
          begin
            k := Trim(Copy(envLines[i], 1, p - 1));
            v := Trim(Copy(envLines[i], p + 1, Length(envLines[i])));
            // Never pre-fill the pairing code (Values[1]) - it's one-time.
            if k = 'SENTRY_BACKEND_URL' then CfgPage.Values[0] := v;
            if k = 'OLLAMA_BASE_URL' then CfgPage.Values[2] := v;
            if k = 'AI_NODE_ID' then nodeId := v;
          end;
        end;
      end;
    end;
    // Already paired? Make it crystal-clear the code is NOT needed again.
    if nodeId <> '' then
      WizardForm.PageDescriptionLabel.Caption :=
        'This machine is ALREADY PAIRED (node ' + nodeId + ').' + #13#10 +
        'Leave the pairing code BLANK and click Next to keep it. ' +
        'To re-pair later use Start Menu -> Pairing - re-pair.';
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = CfgPage.ID then
  begin
    if Trim(CfgPage.Values[0]) = '' then
    begin
      MsgBox('Please enter the Railway backend URL.', mbError, MB_OK);
      Result := False;
    end;
    // Pairing code may be blank on an update (keeps the existing pairing);
    // setup-server.ps1 fails clearly if a fresh node has no code.
  end;
end;

function GetSetupArgs(Param: string): string;
var
  code, tunnel: string;
begin
  Result := '-BackendUrl "' + Trim(CfgPage.Values[0]) + '"' +
            ' -OllamaUrl "' + Trim(CfgPage.Values[2]) + '"';
  code := Trim(CfgPage.Values[1]);
  if code <> '' then
    Result := Result + ' -PairingCode "' + code + '"';
  tunnel := Trim(CfgPage.Values[3]);
  if tunnel <> '' then
    Result := Result + ' -TunnelName "' + tunnel + '"';
end;
