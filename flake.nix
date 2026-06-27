{
  description = "weaver lib";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = [
            pkgs.cabal-install
            pkgs.ghc
            pkgs.stack
            pkgs.git
            pkgs.zlib
            pkgs.pkg-config
            pkgs.python3
            pkgs.python313Packages.pillow
            pkgs.python313Packages.albumentations
            pkgs.python313Packages.numpy
            pkgs.python313Packages.opencv-python
          ];
        };

        packages.default = pkgs.haskellPackages.callCabal2nix "weaver" ./. {};
      });
}

