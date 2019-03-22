# motion_regression
To obtain velocity and acceleration from pose sequences with corresponding timestamp.

## 1. Prerequisites

1. Python 3
2. numpy, pandas, progressbar

__Test environments:__ Python 3.5.2, numpy 1.16.2, pandas 0.24.1, progressbar2 3.39.2

## 2. Usage

'''
python motion_regression.py --pose_path YOUR_PATH 

'''

## 3. Authors

Zhijie Wang (paul dot wangzhijie at outlook dot com)

## 4. References

- Sittel, Florian, Jörg Müller, and Wolfram Burgard. Computing velocities and accelerations from a pose time sequence in three-dimensional space. Technical Report 272, University of Freiburg, Department of Computer Science, 2013.

- [tf.transformation](https://www.lfd.uci.edu/~gohlke/code/transformations.py.html)
