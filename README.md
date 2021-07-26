# cone_detection

## Developer Notes
### HSV ranges

Original calibration values:
(0, 70, 171), (60, 255, 255)

Second iteration values (under the tree- varied brightness):
(0, 100, 171), (180, 255, 255)

Third iteration values(from a distance - bright scene):
(165, 115, 150), (180, 255, 255)


Tips:
- modify Low S first ( saturation gives the best filtering ): 70 - 100
- put High H higher (try maximum): 180
- lower High H to filter bright sky frames