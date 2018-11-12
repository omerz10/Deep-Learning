import numpy as np
import loglinear as ll

STUDENT={'name': 'Omer Zucker_Omer Wolf',
         'ID': '200876548_307965988'}

def classifier_output(x, params):
    """
    Calculate multi level with one hidden layer
    :param x: x
    :param params: W, b, U, b_tag
    :return: vector pf probabilities
    """
    W, b, U, b_tag = params
    hidden = np.tanh(np.dot(x, W) + b)
    mult = np.dot(hidden, U) + b_tag
    probs = ll.softmax(mult)
    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def tanh_derivative(x):
    """
    returns tanh derivative
    :param x: vector
    :return: tanh derivative
    """
    res = 1 - pow(np.tanh(x), 2)
    return res


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    W, b, U, b_tag = params
    softmax_probs = classifier_output(x, params)
    loss = -np.log(softmax_probs[y])
    softmax_probs[y] -= 1

    hidden = np.tanh(np.dot(x, W) + b)
    gb_tag = softmax_probs
    gU = np.outer(hidden, softmax_probs)

    grad_hid = np.dot(U, softmax_probs) * tanh_derivative(hidden)
    gb = grad_hid
    gW = np.outer(x, grad_hid)
    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.


        W = np.random.randn(in_dim, hid_dim)
    b = np.random.randn(hid_dim)

    U = np.random.randn(hid_dim, out_dim)
    b_tag = np.random.randn(out_dim)

    """
    eq1 = np.sqrt(6) / (np.sqrt(hid_dim + in_dim))
    eq2 = np.sqrt(6) / (np.sqrt(hid_dim))

    eq3 = np.sqrt(6) / (np.sqrt(out_dim + hid_dim))
    eq4 = np.sqrt(6) / (np.sqrt(out_dim))

    W = np.random.uniform(-eq1, eq1, [in_dim, hid_dim])
    b = np.random.uniform(-eq2, eq2, hid_dim)
    U = np.random.uniform(-eq3, eq3, [hid_dim, out_dim])
    b_tag = np.random.uniform(-eq4, eq4, out_dim)

    params = [W, b, U, b_tag]
    return params

