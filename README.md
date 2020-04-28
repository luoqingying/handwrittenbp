# handwrittenbp

This is a rough and straight-forward hand-written backward propagation with 2 hidden layers. The whole architecture is 784 - 25 - 20 - 1. We used MNIST data to distinguish between 0 and 1, so the output is binary. We used logistic loss function. For each layer, we used sigmoid as the activation function. We initialized the weights with standard normal distribution. We used the "batch" thing to make learing more efficient. 

Potential problems:

1. The initial pixel values of the images are so large. When we did the matrix multiplication, the resulting numbers are huge and when plugged into the sigmoid function, they will enter the smooth zone and give 1 as a result. If the output number is 1, the log(0) in the loss funciton and gradient will give nan, which stopped the gradient from being passed backward. So we divided all the pixel values by 255 to make them stay in the range of 0 to 1. 

2. We tuned the learning rate to be 18.8, which is uncommon from the usual 0.005 or 0.0005 case because the step size is so tiny that we need a much higher learning rate to make the model learn. Otherwise it will only produce 0 as outcome.

3. Because the log(0) thing in the loss function, we cannot bear the model predict exact 0 or 1, otherwise we will get inf, nan, and thus cannot back-propagate.

4. To keep it simple, we did not add regularization, normalization, but we shuffled the data at the beginning. We did not have so many training samples and the best test accuracy is around 94%. The whole design is not that flexible and we cannot add more layers, but we can still change the learning rate, batch size easily to see how the model fluctuate.
