![image](https://github.com/user-attachments/assets/6610192a-3551-42ca-bcd8-8149ee3b28ac)

# ghost_imaging_demo
 Few code snippets with the tools needed for doing Ghost Imaging simulations: speckle generation, simulation of measurements, recovery, etc.

 Files explanation:
 speckle_generation.py: code to generate speckle patterns. You can change a few parameters to tune how the speckles look. This creates a .h5 file with the speckles, that will be used in the other codes to simulate the measurements and do the recovery.
 
 ghost_demo.py: classical ghost imaging recovery (without compressive sensing). You can choose the number of speckles to use.
 
 compressive_demo.py: Ghost Imaging using different recovery strategies (least squares, l1-minimization, TV-minimization).

 optsim.py: library for optical simulations that I built some time ago (you can find it on Github too). I think I only used a couple simple functions for these simulations (FFTs, etc.)

You can find a longer explanation at: https://fsolt.es/2025/04/easy-ghost-imaging-simulations-and-some-codes-to-do-it-at-home/
