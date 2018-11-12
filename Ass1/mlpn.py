import numpy as np
import loglinear as ll

STUDENT={'name': 'Omer Zucker_Omer Wolf',
         'ID': '200876548_307965988'}

def classifier_output(x, params):
    """"
    Calculate all layers with tanh, last layer with softmax
    params: x, params
    returns: vector of probabilities
    """
    W = params[0]
    b = params[1]

    # calculate first layer
    layer = np.dot(x, W) + b

    # calculate hidden layers
    for i in range(2, len(params), 2):
        layer = np.dot(np.tanh(layer), params[i]) + params[i + 1]

    # calculate last layer
    probs = ll.softmax(layer)
    return probs


def predict(x, params):
    """"
    calculate and return max arg of classifier
    params: x, params
    returns: max argument of the classifier func
    """
    return np.argmax(classifier_output(x, params))


def params_to_couple(params):
    """"
    convert params to couple of b,w
    params: params
    returns: couple of b,w
    """
    couple = [(U, b_tag) for U, b_tag in zip(params[0::2], params[1::2])]
    return couple

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)
    returns:
        loss,[gW1, gb1, gW2, gb2, ...]
    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...
    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    layer = params_to_couple(params)
    cur_layer = [x]
    next_layer = [1]

    for U, b_tag in layer[:-1]:
        next_layer.append(np.dot(cur_layer[-1], U) + b_tag)
        cur_layer.append(np.tanh(np.dot(cur_layer[-1], U) + b_tag))

    y_prime = classifier_output(x, params)
    loss = -np.log(y_prime[y])

    t_vec = y_prime
    t_vec[y] -= 1
    grads = []

    for i, (U, b_tag) in enumerate(reversed(layer)):
        grads.append(t_vec)
        grads.append(np.outer(cur_layer[-i - 1], t_vec))
        tanh_dev = 1 - pow(np.tanh(next_layer[-i - 1]), 2)
        t_vec = np.dot(U, t_vec) * tanh_dev

    grads = list(reversed(grads))
    return loss, grads


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []

    for dim1, dim2 in zip(dims, dims[1:]):
        eq1 = np.sqrt(6) / (np.sqrt(dim1 + dim2))
        eq2 = np.sqrt(6) / (np.sqrt(dim2))
        params.append(np.random.uniform(-eq1, eq1, [dim1, dim2]))
        params.append(np.random.uniform(-eq2, eq2, dim2))

    return params


if __name__ == '__main__':
    from grad_check import gradient_check


    def _loss_and_W_grad(W):
        global b
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[1]

    def _loss_and_U1_grad(U1):
        global W, b, b_tag1, U2, b_tag2
        loss, grads = loss_and_gradients([1,2,3],0,[W, b, U1, b_tag1, U2, b_tag2])
        return loss, grads[2]

    def _loss_and_b_tag1_grad(b_tag1):
        global W, b, U1, U2, b_tag2
        loss, grads = loss_and_gradients([1,2,3],0,[W, b, U1, b_tag1, U2, b_tag2])
        return loss, grads[3]

    def _loss_and_U2_grad(U2):
        global W, b, U1, b_tag1, b_tag2
        loss, grads = loss_and_gradients([1,2,3],0,[W, b, U1, b_tag1, U2, b_tag2])
        return loss, grads[4]

    def _loss_and_b_tag2_grad(b_tag2):
        global W, b, U1, b_tag1, U2
        loss, grads = loss_and_gradients([1,2,3],0, [W, b, U1, b_tag1, U2, b_tag2])
        return loss, grads[5]

    for _ in xrange(10):
        W, b, U1, b_tag1, U2, b_tag2 = create_classifier([3,5,7,9])

        gradient_check(_loss_and_b_tag2_grad, b_tag2)
        gradient_check(_loss_and_U2_grad, U2)
        gradient_check(_loss_and_b_tag1_grad, b_tag1)
        gradient_check(_loss_and_U1_grad, U1)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
