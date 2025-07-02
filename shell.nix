# nix-direnv file
{ pkgs ? import <nixpkgs> {}}:

pkgs.mkShell {
  packages = [
    pkgs.portaudio
    pkgs.pkg-config
    pkgs.gcc
    pkgs.python3
  ];
}
