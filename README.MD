# motion_regression (Update 23/3/19)

__What's New:__

- Support multiple pids processing.

- Remove progressbar (bugs exist)

---

To obtain velocity and acceleration from pose sequences with corresponding timestamp.

<img src="https://github.com/paulwong16/motion_regression/blob/master/cp.png" alt="compare" />

The result is smoother comparing to the EuRoc estimated groundtruth. (V2_01_easy)

## 1. Prerequisites

1. Python 3
2. numpy, pandas, ~~progressbar~~

__Test environments:__ Python 3.5.2, numpy 1.16.2, pandas 0.24.1, ~~progressbar2 3.39.2~~

## 2. Usage

```
python motion_regression.py --pose_path YOUR_PATH
```

## 3. Authors

Zhijie Wang (paul dot wangzhijie at outlook dot com)

## 4. References

- Sittel, Florian, Jörg Müller, and Wolfram Burgard. Computing velocities and accelerations from a pose time sequence in three-dimensional space. Technical Report 272, University of Freiburg, Department of Computer Science, 2013.

- Burri, Michael, et al. "The EuRoC micro aerial vehicle datasets." The International Journal of Robotics Research 35.10 (2016): 1157-1163.

- [tf.transformation](https://www.lfd.uci.edu/~gohlke/code/transformations.py.html)
