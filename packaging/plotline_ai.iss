; Plotline-AI installer (Inno Setup 6). Build: packaging\build_installer.bat
; Per-user install (no administrator rights): %LOCALAPPDATA%\Programs\Plotline-AI

#define AppName "Plotline-AI"
#define AppVersion "2.0"
#define Root "D:\Plotline-AI"
#define RuntimeEnv "C:\Users\nt01173\micromamba\envs\roadmarking-ai"
#define Model "models\mask2former_swin_s\production\v2_realistic_v1_fix_step0500"

[Setup]
AppId={{5E88E294-F8AE-42A3-B803-FDBA13B6BF2B}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Plotline-AI
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#Root}\dist
OutputBaseFilename=Plotline-AI-{#AppVersion}-Setup
SetupIconFile={#Root}\src\roadmarking\app\assets\plotline_ai.ico
UninstallDisplayIcon={app}\plotline_ai.ico
UninstallDisplayName={#AppName} {#AppVersion}
Compression=lzma2/normal
SolidCompression=no
LZMANumBlockThreads=8
LZMAUseSeparateProcess=yes
DiskSpanning=yes
DiskSliceSize=max
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
VersionInfoVersion={#AppVersion}.0.0
VersionInfoDescription=Plotline-AI automatic LiDAR road-marking digitization

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
; Python runtime (complete conda environment: CUDA PyTorch, GDAL, PDAL, ...)
Source: "{#RuntimeEnv}\*"; DestDir: "{app}\runtime"; Flags: recursesubdirs createallsubdirs ignoreversion; Excludes: "__pycache__\*,*.pyc,\pkgs\*,\conda-meta\history"
; application code (locked 1.2), settings, model, client reference marks, guide
Source: "{#Root}\src\*"; DestDir: "{app}\src"; Flags: recursesubdirs createallsubdirs ignoreversion; Excludes: "__pycache__\*,*.pyc,*.egg-info\*"
Source: "{#Root}\configs\*"; DestDir: "{app}\configs"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#Root}\{#Model}\*"; DestDir: "{app}\{#Model}"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#Root}\data\raw\annotations\Highway\*.gpkg"; DestDir: "{app}\data\raw\annotations\Highway"; Flags: ignoreversion
Source: "{#Root}\data\raw\annotations\Interchange\*.gpkg"; DestDir: "{app}\data\raw\annotations\Interchange"; Flags: ignoreversion
Source: "{#Root}\docs\Plotline-AI_User_Guide.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "{#Root}\packaging\launch.pyw"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\src\roadmarking\app\assets\plotline_ai.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\launch.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\plotline_ai.ico"; AppUserModelID: "Plotline-AI.Desktop"
Name: "{group}\Plotline-AI User Guide"; Filename: "{app}\docs\Plotline-AI_User_Guide.md"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\launch.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\plotline_ai.ico"; AppUserModelID: "Plotline-AI.Desktop"; Tasks: desktopicon

[Run]
Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\launch.pyw"""; WorkingDir: "{app}"; Description: "Start {#AppName}"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\src"
