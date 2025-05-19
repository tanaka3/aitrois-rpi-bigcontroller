If you use multiple cameras, you will need the ``camera number`` and  the ``camera id``.

To find the camera ID, run the command ``libcamera-hello --list-cameras``.

#### example
```
> libcamera-hello --list-cameras
Available cameras
-----------------
0 : imx500 [4056x3040 10-bit RGGB] (/base/axi/pcie@120000/rp1/i2c@88000/imx500@1a)]
    Modes: 'SRGGB10_CSI2P' : 2028x1520 [30.02 fps - (0, 0)/4056x3040 crop]
                             4056x3040 [10.00 fps - (0, 0)/4056x3040 crop]

1 : imx500 [4056x3040 10-bit RGGB] (/base/axi/pcie@120000/rp1/i2c@80000/imx500@1a)
    Modes: 'SRGGB10_CSI2P' : 2028x1520 [30.02 fps - (0, 0)/4056x3040 crop]
                             4056x3040 [10.00 fps - (0, 0)/4056x3040 crop]
```

```
from picamera2 import Picamera2

cameras = Picamera2.global_camera_info()
for i, cam in enumerate(cameras):
    print(f"{cam}")
```


The first number is the camera number.

This ID is used as an argument for Picamera2.

```
picam = Picamera2(camera_num=0)
```

``/base/axi/pcie@120000/rp1/i2c@80000/imx500@1a`` is the camera id.

This is the argument for IMX500.

```
imx500 = IMX500(args.model, "/base/axi/pcie@120000/rp1/i2c@80000/imx500@1a")
```

Please use it according to your environment from now on.
