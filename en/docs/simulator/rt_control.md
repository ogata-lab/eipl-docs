# Realtime Motion Generation
Move to the simulator folder and run the realtime motion generation program `6_rt_control.py` with the weight file as an argument. This program repeats 10 times the object grasping operation placed at random positions.

```bash
$ cd ../simulator/
$ python3 ./bin/6_rt_control.py ../sarnn/log/YEAR_DAY_TIME/SARNN.pth
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/ito/var/robosuite/robosuite/scripts/setup_macros.py (macros.py:55)
[1/10] Task succeeded!
[2/10] Task succeeded!
```