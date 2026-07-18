{
  description = "NixClaw agent, benchmarks, and shared integration assets";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs, ... }:
    let
      systems = [
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      packagesFor = system: import nixpkgs { inherit system; };
      appNames = {
        agent = "nixclaw-agent";
        adversarial = "nixclaw-adversarial";
        bench = "nixclaw-bench";
        fixture = "nixclaw-fixture";
        policy = "nixclaw-policy";
        scorecard = "nixclaw-scorecard";
      };
    in
    {
      overlays.default = final: _prev: {
        nixclaw = final.callPackage ./nix/package.nix { };
      };

      packages = forAllSystems (
        system:
        let
          pkgs = packagesFor system;
          nixclaw = pkgs.callPackage ./nix/package.nix { };
        in
        {
          inherit nixclaw;
          default = nixclaw;
        }
      );

      apps = forAllSystems (
        system:
        builtins.mapAttrs (_name: program: {
          type = "app";
          program = "${self.packages.${system}.nixclaw}/bin/${program}";
          meta.description = "Run ${program}";
        }) appNames
        // {
          default = {
            type = "app";
            program = "${self.packages.${system}.nixclaw}/bin/nixclaw-agent";
            meta.description = "Run nixclaw-agent";
          };
        }
      );

      checks = forAllSystems (system: {
        package = self.packages.${system}.nixclaw;
      });

      formatter = forAllSystems (system: (packagesFor system).nixfmt-tree);
    };
}
