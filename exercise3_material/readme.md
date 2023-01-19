## Regularization and the Recurrent Layer

### Regularizers
***Purpose***: Preventing overfitting by adding a penalty term to the loss function. It can also help to improve the generalization ability of the model by reducing the complexity of the model. 

- L1 Regularizer: Adds the *absolute values* of the model weights  
- L2 Regularizer: Adds the *square values* of the model weights  
In general L1 regularization is more useful when we have a huge number of features and we need to select only a subset of them, while L2 regularization is more commonly used to prevent overfitting by trying to keep the weight values small.
- Droupout: Randomly droppings out neurons during training, the network is forced to learn multiple independent representations of the input, making it less prone to overfitting and more robust to changes in the input data.

- Early Stopping

- Data Augmentation

### Batch Normalization 
***Purpose***: Calculating the mean and standard deviation of the activations for each mini-batch and changing in the distribution of the inputs. BN helps to stabilize the training by reducing the internal covariate shift and smooths the loss landscape.

### Activation Functions
- TanH: $tanh(x)$

- Sigmod: $\frac{1}{1+e^-x}$

### RNN
- Forwward: 
h_t = tanh(xt_hat * W_h)
y_t = sigmod(h_t * W_hy)

- Backward
loop the reversed layer, and get the gradient of the backward functions.