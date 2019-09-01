<div style="flex-layout: row;">
  <img src="images/title.png" />
  <img src="images/confusion.jpg" height=100px />
</div>

---

Demo of blackbox & gradient-free optimisation of NN 


## Installation
---

See https://github.com/pybox2d/pybox2d/blob/master/INSTALL.md for installation of SWIG


```bash
$ git clone https://github.com/JonasRSV/random-direction-ascent.git
$ cd random-direction-ascent
$ pip3 install -r requirements.txt
``` 


## Run any of the environments 
---

```bash
$ python3 (mountaincar.py | lunar-lander.py | pendelum.py | cartpole.py)
```


|environment   | score  | iteration  | score to win environment  | time |
|---|---|---|---|---|
| mountain car   | -94  | 95  | > -200 | 0.7 seconds |
| lunar lander   | 237  | 126  | > 200  | 50 seconds |
| cartpole  |  224  | 17  | 200 | 0.17 seconds |
