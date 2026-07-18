{
  lib,
  python312Packages,
}:

python312Packages.buildPythonApplication {
  pname = "nixclaw";
  version = "0.1.0";
  pyproject = true;

  src = lib.cleanSource ../.;

  build-system = with python312Packages; [ hatchling ];

  dependencies = with python312Packages; [
    httpx
    jsonschema
    pydantic
    pyyaml
    typer
  ];

  nativeCheckInputs = with python312Packages; [ pytestCheckHook ];
  pythonImportsCheck = [ "nixclaw" ];

  postInstall = ''
    mkdir -p "$out/share/nixclaw"
    cp -r schemas skills openshell workloads "$out/share/nixclaw/"
  '';

  passthru = {
    schemaDir = "share/nixclaw/schemas/nixclaw/v1";
    skillDir = "share/nixclaw/skills/nixclaw";
    openShellPolicy = "share/nixclaw/openshell/policy.yaml.in";
    workloadDir = "share/nixclaw/workloads";
  };

  meta = {
    description = "Contained recursive optimization for NixOS-hosted Hermes agents";
    homepage = "https://github.com/benthecarman/nixclaw";
    mainProgram = "nixclaw-agent";
    platforms = lib.platforms.linux;
  };
}
