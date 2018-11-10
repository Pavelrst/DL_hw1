r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
No. Generalizing does not necessarily increase together with increasing the k value in Knn. Increasing it does increase generalization but uptil a certain limit which varies with the model required to describe and the examples. An extreme example is that setting up k to be the same as the entire train set, probably wouldn't get very good results, since it will allways give the same answer which is the label that is most common in the training set. In the other extreme setting up k to be one would be extreamly sensitive to noisy data and outliers.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
For each delta > 0 we will give some penalty to a missclasified classification. So the optimization of the specific size of the delta, is really quite flexbile and varies with our prefernces and the size differences of the weight in the weight matrix and order of the missclassification events.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**

1.The linear model is learning to indentify digits. The errors it made are caused because of the digit written poorly. But we as humans, that write those digits by pan regularly, are accostumbred to sometimes miss part of the letter, and altough the digit is not complete already recognize the movement pattern (if the writting make any sense, whats or ever).

2. WE believe that KNN would have a more difficult time recognizing this digits. Since the liner classifer relies on the lines, directions, positions and relation to each other, KNN would expect to all same digits to look a like and that often ain't the cases since different person write differently or sometimes even the same person writes differently! (size, shape, location, etc.) What matters most are the lines of the digits and the dynamics between them.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**
Too low/good. It is nice it keep improving and yields result which are not bad at all. **BUT**, it doesn't reach an extremum point and just keep moving monotoniclly, what means the learning rate wasn't sufficient. If it was good, we couldv'e through the middle/end, reaching the extremum, optimal point and then getting worst result. If it was too high, we would pass the extremum point by over shoot and wouldn't get it with good resolution/accuracy. But, it is good, because the results are good, and the derivative is very near zero by the end (altough it doesn't converge, and doesn't really, "gets there")

Slightly underfitting. For the same reason as above, we belive that the model is a bit underfitting (but good overall) because if the learning rate was too high, it would be overfitting, and the results would be accordingly. since, it doesn't we put it on the lower half, but really only a bit below.



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**The ideal pattern to see in a residual plot are points dispersed randomly around the horizontal axis. 
Based on the residual plots wev'e got we can say that the linear regression model is fit to represent this model. 
The top 5 plot in the other hand wasn't great and does not represent properly the model.
"""

part4_q2 = r"""
**Your answer:** When choosing Lambda it was more important to find it roughly and the correct order of magnitude. the exact value is not very important. Also, without having an order of magnitude looking for it a linear way with high precision (since we started with a very low number and jumps) would require many iterations.

$Lambdarange * Degreerange = 20*3 = 60$


"""

# ==============
