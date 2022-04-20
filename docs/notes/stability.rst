.. _stability:

Stability
=========
Stability is a key package issue. SoulsGym converts Dark Souls III into a gym environment by reading
and writing the process memory. We have no direct API for the game and therefore no guarantees
for the game's behavior. All memory locations have been found by memory scans, pointer graph
analysis and a lot of try and error. The game remains a semi-blackbox to us, so it's hard to iron
out all the issues without extensive testing. If you do encounter any bugs please notify us to help
us improve stability!