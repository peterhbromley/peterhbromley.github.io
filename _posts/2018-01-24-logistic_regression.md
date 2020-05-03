---
layout: post
title: Logistic Regression
subtitle: Implementation and Visualization
date: 2018-01-24
background: '/img/posts/lr/3d-data.png'
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### Description

Logistic Regression is a supervised learning algorithm used to predict a given input's class based on it's features. While *linear* regression can be used to predict continuous variables such as housing prices, *logistic* regression is a much more effective algorithm when dealing with discrete output, such as predicting whether or not an email is spam. Logistic regression is useful for solving "classification" problems, while linear regression is useful for solving "regression" problems.

We can imagine a dataset composed of an $$M$$x$$N$$ matrix, $$x$$, and a length $$M$$ vector, $$y$$. $$x$$ consists of length $$N$$ feature vectors containing quantitative information about certain attributes for each data point. For example, let's say we are trying to predict whether or not a student will pass an exam. Features (also called parameters) could be hours studied, hours of sleep, GPA, etc.  The output vector $$y$$ tells us the class that the corresponding input data point belongs to, which in this case is either passed or didn't pass. Optimally, we would have a large dataset with information about other students and their "attributes", and whether or not they passed the exam. Logistic regression is a supervised learning algorithm, meaning we must know the outcome of all of the data in our dataset before running the algorithm.

For making a classification prediction, we need a function that maps our input $$x$$ to a binary (in our case) output space $$y$$. This function is called a hypothesis. Linear regression uses the linear model, $$\theta^{T}x$$ (where $$\theta$$ is a vector of weights and $$x$$ is an input feature vector), to make a prediction. However, we want a non-linear output to classify our predictions into a binary space, so we will use a __link function__ to achieve this. We will use the logit function: $$log(\frac{p}{1-p})$$ where $$p$$ is the probability given to a variable, $$\frac{p}{1-p}$$ is the odds based on that probability, and logit($$p$$) is thus the *log-odds*.

We can then relate the linear model to the link function: $$log(\frac{p}{1-p}) = \theta^{T}x$$, solve for $$p$$, and we get $$p = \frac{1}{1+e^{-\theta^{T}x}}$$. The right side of this equation is the __logistic function__, and it is what we will use as our hypothesis for logistic regression. The logistic function is a sigmoid curve because as we can see when we plot it (with only one input variable), it is S-shaped:



```python
def sigmoid(z):
    return 1/(1+np.exp(-z))

x = np.arange(-6, 6, 0.1)
y = np.array([sigmoid(num) for num in x])
plt.plot(x, y)
plt.show()
```

| ![sigmoid.png]({{ site.baseurl }}/img/posts/lr/sigmoid.png)
|:--:|
| *Logistic function is constrained between y = 0 and y = 1 for any input x* |

So, if we plug our linear model into this sigmoidal function, for any weights and data points that we feed our model we will get an output between $$0$$ and $$1$$. Thus, if we have two classes that we are trying to predict, we can assign a $$0$$ to one class and a $$1$$ to another (e.g. $$0 =$$ not spam, $$1 =$$ spam). We can interpret this output as the probability that, given input $$x$$, the output $$y$$ will be $$1$$.

Now we have a function that maps input to output, but we need a way to calculate the weights, $$\theta$$, if we want to make accurate predictions. Since our data is all labeled (we know the outcome of the inputs), we can diagnose the predictive power of our current weight vector by mapping our inputs and comparing the calculated output (predictions) to the actual outcomes, $$y$$. Our hope is to match them perfectly, though this is unrealistic.

We can use a __likelihood function__ to generalize this diagnosis. Given a function that, if the likelihood is low we have a poor predictor, and if the likelihood is high we have a good predictor, we obviously would want to maximize that function. We can use the *log likelihood* for this, which is the following (Note: subscript $$i$$ means $$i^{th}$$ data pt in dataset):

$$ll(\theta) = \sum_{i=0}^{M} y_{i}log(sig(\theta^{T}x_{i})) + (1 - y_{i})log(1 - sig(\theta^{T}x_{i}))$$

I will not include the derivation of this function in this writeup, but [here is a link if you are interested](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/pdfs/40%20LogisticRegression.pdf). The log likelihood is pretty intuitive when broken down. Recall that our prediction will be made by $$sig(\theta^{T}x_{i})$$, and that $$log(1)$$ is $$0$$, and log(0) is negative infinity. Also notice that if $$y_i$$ is $$1$$ the second term is automatically 0 (and thus ignored), and vice versa if $$y_i$$ is $$0$$. Furthermore, if $$y_i$$ is 1 and we incorrectly predict a number close to zero, we get we a large negative value. If $$y_i$$ is 1 and we predict a number close to 1, we get a very small negative value. The same concepts hold for when $$y_i$$ is 0 and we are concerned with the second term. Add all of these values up across the dataset, and we have a single number that describes the effectiveness of our current weights.

We can simplify (write out the sig functions and it will make sense) to get:

$$ll(\theta) = \sum_{i=0}^{M} y_{i}\theta^{T}x_{i} - log(1 + e^{\theta^{T}x_{i}})$$

So we want to maximize this cost function. Normally we do this by taking the derivative and calculating the maximum. But we can't actually calculate a closed form solution in this case, so we have to use an algorithm called stochastic gradient descent to find our weights. We take the gradient of $$ll(\theta)$$ and then step a small amount in that direction (because the gradient points in the direction of the greatest rate of increase), then we recalculate the gradient at the new location, and so on.

Taking the gradient of $$ll(\theta)$$ is equivalent to taking the partial derivative with respect to $$\theta$$. This leaves us with:

$$\frac{\partial ll(\theta)}{\partial \theta_j} = \sum_{i=0}^{M} (y_{i} - sig(\theta^{T}x_{i}))x_{ij}$$

Note: the subscript $$j$$'s show that this operation is performed on each individual parameter of $$\theta$$ to get the gradient.

Now we have the gradient, but we still need a way to step up to the maximum. We want to do this by updating our weights incrementally using the gradient. The process is simple: for all $$j$$ parameters in $$\theta$$, update by performing:

$$\theta_j = \theta_j + r\frac{\partial ll(\theta)}{\partial \theta_j}$$

where $$r$$ is a very small number. By doing this, our log likelihood is bound to increase and we get closer to the maximum with every iteration. As we step closer to the maximum the difference between log likelihood calculations across sequential iterations will converge to inconsequential values, at which point we can be confident that we have essentially hit the maximum.

Now we have a theta that is optimized for prediction, and we can predict new inputs using the sigmoid function and our calculated weights. We can test just how powerful our model is using various methods that I will describe later.


### Implementation

For this implementation, I am going to generate a "fake" dataset using fixed weights and the sigmoid function. Then I will initialize a new weight vector, run the model, and compare the weights that the model calculated and compare them to the fixed weights used to generate the dataset. The inputs will be two-dimensional (three counting the bias unit).

To generate the dataset, I will generate a random point, calculate the logistic function with the point and the given theta (the target weights), and perform a "weighted coin flip" with the value to determine the output. For example, if the logistic function applied to a point $$x$$ and a given weight vector $$\theta$$ evaluates to 0.6, the weighted flip gives a $$60$$ percent chance of $$y=1$$ and a $$40$$ percent chance of $$y=0$$. With this method, my dataset will (very likely) NOT be linearly separable, but will give my algorithm a chance to find accurate weights. Here are the functions:

```python
def weighted_flip(prob):
    return random.random() < prob

def sigmoid(z):
    return 1/(1+np.exp(-z))

def generate_data(nb_pts, theta_given):
  x = np.ones((nb_pts, 3))      # x[0] all ones for bias
  y = np.ones(nb_pts)
  for i in range(x.shape[0]):
      x[i][1] = random.uniform(-5, 5)
      x[i][2] = random.uniform(-5, 5)
      prob = sigmoid(np.dot(x[i], theta_given))
      y[i] = float(weighted_flip(prob))
  return x, y
```

So if we generate a dataset with 10000 points and a target $$\theta = ({-1}, 1)$$, our dataset looks like this:

| ![data.png]({{ site.baseurl }}/img/posts/lr/data.png)
|:--:|
| *Generated "fake" dataset with $$\theta = ({-1}, 1)$$, yellow pts are $$y=1$$, purple pts are $$y=0$$* |

Even though we don't actually need to calculate the log-likelihood to solve for $$\theta$$, it is a good way to track the progress of our algorithm when we run it. We can define it like this:

```python
def log_likelihood(x, y, theta):
    return np.sum(y * np.dot(x, theta) - np.log(1 + np.exp(np.dot(x, theta))))
```

And finally, we need a function to update our weights on each iteration using the gradient of the log-likelihood. We need to pass lr (the learning rate) here:

```python
def update_weights(x, y, theta, lr):
    hyp = sigmoid(np.dot(x, theta))
    update = lr * np.dot(x.transpose(), y - hyp)
    return theta + update
```

Now we can write a function that runs logistic regression with stochastic gradient descent. We start with a random weight vector theta, iterate for the desired number of iterations, and on each iteration we update theta. In my implementation I print the log-likelihood once in a while (if print_ll=True) so that I can see that the algorithm is working, but I don't have thousands of lines of output. You can ignore the saved_thetas variable; it is just used for a later visualization.

```python
def run(x, y, lr=0.0001, nb_iter=50000, save_thetas=False, print_ll=False):

    # initialize random starting weights
    theta = np.array([random.uniform(-5, 5) for i in range(x.shape[1])])
    saved_thetas = [theta]

    for i in range(nb_iter):
        theta = update_weights(x, y, theta, lr)

        if print_ll:
            if i % (nb_iter * 0.05) == 0:
                print(log_likelihood(x, y, theta))

        if save_thetas:
            saved_thetas.append(theta)

    if save_thetas:
        return saved_thetas
    else:
        return theta
```

Let's run the model with a target weight vector of $$({-1}, 1)$$:

```python
theta_given = np.array([0, -1, 1])      # 3 dimensional because bias
x, y = generate_data(10000, theta_given)
theta = run(x, y, lr=0.000001, save_thetas=False, print_ll=True)
print(theta)
```

Our output is:

```python
Starting theta: [-3.31548264  2.31914421  4.43705435]
-45895.04341909778
-3345.6371296113034
-2791.8770315328534
-2764.8100302256307
-2764.4596533025665
-2764.4556864035117
-2764.4556378278244
...
Calculated theta: [-0.04165553 -1.00920975  1.01210264]
```

(I'm only showing the first seven prints because it has essentially converged at that point)

Our calculated theta is clearly very close to our target theta, so we are happy! The algorithm is working as it is supposed to.


### Visualization

We can visualize the process of logistic regression with two parameters in a three dimensional space. Let's now plot our data like we did before (with the $$x$$ and $$y$$ axes corresponding to features 1 and 2 of a given input), but add in the criteria that a point's $$z$$ axis value corresponds to its predetermined output. So if an input $$(1, 2)$$ was assigned $$y = 1$$ by our dataset generation, then we would plot it as $$(1, 2, 1)$$. We then get a 3D graph like this:

| ![3d-data.png]({{ site.baseurl }}/img/posts/lr/3d-data.png)
|:--:|
| *Generated dataset in three dimensions* |

We can also plot our hypothesis function in three dimensions, using the equation: $$z = \frac{1}{1 + e^{-(\theta_{1}x + \theta_{2}y)}}$$. Using $$\theta = (1, 1)$$ we get this:

| ![3d-sigmoid.png]({{ site.baseurl }}/img/posts/lr/3d-sigmoid.png)
|:--:|
| *Sigmoid curve in three dimensions* |

If we go to a point in the x-y plane describing our input, we can then draw an orthogonal line up until we intersect the curve, at which point the $$z$$ value is the *probability that our hypothesis will classify that input as a 1*. Let's use the input point $$(-4, -4)$$ as an example. If we go to $$(-4, -4)$$ in the x-y plane, we can see that the curve is very close to $$z = 0$$ at that point, meaning our current hypothesis will almost certainly not classify $$(-4, -4)$$ as a 1.

So, given the two previous images, it is clear that we want our final hypothesis curve to "fit" our dataset as well as possible. This means we want the part of the curve that is close to $$z = 1$$ to align with our data points classified as 1's, and the part of the curve close to $$z = 0$$ to align with our data points classified as 0's. We can see this fitting process happen as we run logistic regression using the following animation:

| ![lr-animation.gif]({{ site.baseurl }}/img/posts/lr/lr-animation.gif)
|:--:|
| *Sigmoid curve plotted repeatedly over initial iterations of logistic regression* |

Our hypothesis starts out completely random, so the curve is completely unaligned with our dataset. This results in a lot of misclassifications and thus a very small (large negative) log likelihood. As gradient descent improves our hypothesis with every iteration, the curve starts to fit better and better to the dataset until it converges. Eventually, it fits almost exactly how we want it to.
