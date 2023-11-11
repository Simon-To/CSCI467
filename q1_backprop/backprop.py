"""Code for HW3 Problem 1: Backpropagation."""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit as sigmoid

OPTS = None

EPSILON = 1e-8  # epsilon to use for numerical gradient computation

# A linearly separable classification dataset
SIMPLE_DATASET = [
    (np.array([2.0, 2.0]), 1),
    (np.array([1.0, 0.0]), 1),
    (np.array([1.0, 2.0]), -1),
    (np.array([0.0, -1.0]), -1),
]

# The XOR dataset, which is not linearly separable
XOR_DATASET = [
    (np.array([0.0, 0.0]), -1),
    (np.array([1.0, 1.0]), -1),
    (np.array([1.0, 0.0]), 1),
    (np.array([0.0, 1.0]), 1),
]

### Start of (modified) backprop lecture demo code ###

class Node(object):
    """Node in a computation graph.

    Must store the following:
        - self.value: Numerical value, computed during forward pass
        - self.grad: Gradient, computed during backward pass
        - self.topo_order: Topologically-sorted list of nodes
    """
    def backward(self):
        """Does backward pass.

        Assumption: self.grad has been set by parent nodes already.
        Propagates this gradient to all child nodes.
        """
        raise NotImplementedError

    def compute_grad(self):
        # This function only gets called on the output node.
        self.grad = np.float64(1)   # Gradient of output w.r.t itself is 1
        for node in self.topo_order[::-1]:  # Reverse the list
            # Call backward() in reverse topological order
            node.backward()


class InputNode(Node):
    def __init__(self, value, topo_order):
        self.value = value
        if isinstance(value, np.ndarray):
            # Initialize self.grad to an array of all zeros with the same shape as value
            self.grad = np.zeros_like(value, dtype=np.float64)
        else:
            self.grad = np.float64(0)
        self.topo_order = topo_order
        topo_order.append(self)

    def backward(self):
        pass  # Nothing to do, as there are no child nodes

class AddNode(Node):
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.value = n1.value + n2.value
        if isinstance(self.value, np.ndarray):
            # Initialize self.grad to an array of all zeros with the same shape as value
            self.grad = np.zeros_like(self.value, dtype=np.float64)
        else:
            self.grad = np.float64(0)
        self.topo_order = n1.topo_order
        self.topo_order.append(self)

    def backward(self):
        # Recursively, self.grad has gradient of output w.r.t. this node
        # Changing n1 changes (n1 + n2) by the same amount
        # Same for n2
        # Thus gradient w.r.t. n1 and n2 is the same as gradient w.r.t this node
        self.n1.grad += self.grad
        self.n2.grad += self.grad

class MulNode(Node):
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.value = n1.value * n2.value
        self.grad = np.float64(0)
        self.topo_order = n1.topo_order
        self.topo_order.append(self)

    def backward(self):
        self.n1.grad += self.grad * self.n2.value
        self.n2.grad += self.grad * self.n1.value

class PowerNode(Node):
    def __init__(self, n, power):
        # n is a Node, power is just a float/integer
        self.n = n
        self.power = power
        self.value = n.value**power
        self.grad = np.float64(0)
        self.topo_order = n.topo_order
        self.topo_order.append(self)

    def backward(self):
        self.n.grad += self.power * self.n.value**(self.power-1) * self.grad

class ReluNode(Node):
    def __init__(self, n):
        self.n = n
        self.value = np.maximum(n.value, 0)
        self.topo_order = n.topo_order
        self.topo_order.append(self)
        if isinstance(self.value, np.ndarray):
            # Initialize self.grad to an array of all zeros with the same shape as value
            self.grad = np.zeros_like(self.value, dtype=np.float64)
        else:
            self.grad = np.float64(0)

    def backward(self):
        self.n.grad += self.grad * (self.value > 0)

### End of Backprop lecture demo code ###

class ConstantMulNode(Node):
    """Node that computes a * x for some constant a."""
    def __init__(self, n, a):
        self.n = n
        self.a = a
        self.value = a * n.value
        self.grad = np.float64(0)
        self.topo_order = n.topo_order
        self.topo_order.append(self)

    def backward(self):
        self.n.grad += self.grad * self.a

class LogNode(Node):
    """Node that computes log(x), where x is the value of node n."""
    def __init__(self, n):
        self.n = n
        self.topo_order = n.topo_order
        self.topo_order.append(self)
        # TODO: Set self.value (forward pass) and initialize self.grad    (DONE)
        ### BEGIN_SOLUTION 1b
        # Set self.value (forward pass)
        self.value = np.log(n.value)
        # Initialize self.grad
        self.grad = np.float64(0)
        ### END_SOLUTION 1b

    def backward(self):
        ### BEGIN_SOLUTION 1b
        # TODO: (DONE)
        self.n.grad += self.grad * np.divide(np.float64(1), self.n.value)
        ### END_SOLUTION 1b

class SigmoidNode(Node):
    """Node that computes sigmoid(x), where x is the value of node n."""
    def __init__(self, n):
        self.n = n
        self.topo_order = n.topo_order
        self.topo_order.append(self)
        # TODO: Set self.value (forward pass) and initialize self.grad  (DONE)
        ### BEGIN_SOLUTION 1e
        # Set self.value (forward pass)
        self.value = sigmoid(n.value)
        # Initialize self.grad
        self.grad = np.float64(0)
        ### END_SOLUTION 1e

    def backward(self):
        ### BEGIN_SOLUTION 1e
        # TODO: (DONE)
        # u = sigmoid(self.n.value)
        self.n.grad += self.grad * (self.value * (np.float64(1) - self.value))
        ### END_SOLUTION 1e

class DotNode(Node):
    """Node that computes x^T v.

    n1 is a node whose value is a vector x.
    n2 is a node whose value is a vector v.
    """
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.topo_order = n1.topo_order
        self.topo_order.append(self)
        # TODO: Set self.value (forward pass) and initialize self.grad      (DONE)
        ### BEGIN_SOLUTION 1g

        # Set self.value (forward pass)
        self.value = np.dot(n1.value, n2.value)
        # Initialize self.grad
        self.grad = np.zeros_like(n1, dtype=np.float64)
        ### END_SOLUTION 1g

    def backward(self):
        ### BEGIN_SOLUTION 1g
        # TODO: (DONE)
        self.n1.grad += np.dot(self.grad, self.n2.value)
        self.n2.grad += np.dot(self.grad, self.n1.value)
        ### END_SOLUTION 1g

class MVMulNode(Node):
    """Node that computes the matrix-vector product Mv.

    M is a node whose value is a matrix of shape (p, d).
    v is a node whose value is a vector of length d.
    """
    def __init__(self, M, v):
        p, d = M.value.shape
        self.M = M  # p x d
        self.v = v  # d
        self.topo_order = M.topo_order
        self.topo_order.append(self)
        # TODO: Set self.value (forward pass) and initialize self.grad
        ### BEGIN_SOLUTION 1k
        # Set self.value (forward pass)
        self.value = np.dot(M.value, v.value)
        # Initialize self.grad
        self.grad = np.zeros(p, dtype=np.float64)
        ### END_SOLUTION 1k

    def backward(self):
        ### BEGIN_SOLUTION 1k
        p, d = self.M.value.shape
        flattened_g = self.grad.reshape(p, 1)
        flattened_v = self.v.value.reshape(1, d)
        self.M.grad += np.dot(flattened_g, flattened_v)
        self.v.grad += np.matmul(self.M.value.T, self.grad)
        ### END_SOLUTION 1k

def gradient_check_1(a_test, b_test, c_test):
    """Run gradient check for LogNode, SigmoidNode, and DotNode."""
    # First let's define a helper function.
    # Given numpy arrays a_val, b_val, and c_val, this function:
    #     - Creates a computation graph that computes f(a_val, b_val, c_val)
    #     - Returns the output node of this computation graph
    def compute_f(a_val, b_val, c_val):
        """Helper function for gradient checking.

        Args: numpy array values for a, b, and c
        Returns: 2-tuple containing:
            - input_nodes: list of InputNode objects corresponding to a, b, and c
            - output_node: The output node whose value is f(a_val, b_val, c_val)
        """
        # Create the shared topo_order list and InputNode's
        topo_order = []
        a = InputNode(a_val, topo_order)
        b = InputNode(b_val, topo_order)
        c = InputNode(c_val, topo_order)
        input_nodes = [a, b, c]

        # Now, create the computation graph that computes f
        output_node = None  # TODO: Set this so we can return it    (DONE)
        ### BEGIN_SOLUTION 1h
        # Step 1: getting the dot products:
        a_dot_b = DotNode(a, b)
        a_dot_c = DotNode(a, c)

        # Step 2: log and sigmoid:
        log_a_dot_b = LogNode(a_dot_b)
        sigmoid_a_dot_c = SigmoidNode(a_dot_c)

        # Step 3: 3 * sigmoid(a, c):
        x_neg_3_sigmoid_a_dot_c = ConstantMulNode(sigmoid_a_dot_c, np.float64(-3))

        # Step 3: final output:
        output_node = AddNode(log_a_dot_b, x_neg_3_sigmoid_a_dot_c)


        ### END_SOLUTION 1h
        return input_nodes, output_node

    # Compute gradients with backpropagation
    (a, b, c), out = compute_f(a_test, b_test, c_test)
    print(f'Computed value of f(a_test, b_test, c_test): {out.value}')
    out.compute_grad()  # Runs backpropagation, setting a.grad, b.grad, and c.grad

    # Now, numerically compute the gradient by changing each coordinate of each input by EPSILON
    # and observing how much f changes.
    # TODO: store the numerical gradients in the variables below.   (DONE)
    a_grad_numerical = np.zeros(2)  # Numerically computed gradient with respect to a
    b_grad_numerical = np.zeros(2)  # Numerically computed gradient with respect to b
    c_grad_numerical = np.zeros(2)  # Numerically computed gradient with respect to c
    ### BEGIN_SOLUTION 1h

    # Evaluate function at (a, b, c)
    orig = out.value

    # Measure change in output when you change each input by EPSILON
    # (a_a, b_a, c_a), out_a = compute_f((a_test + np.float64(EPSILON)), b_test, c_test)
    # (a_b, b_b, c_b), out_b = compute_f(a_test, (b_test + np.float64(EPSILON)), c_test)
    # (a_c, b_c, c_c), out_c = compute_f(a_test, b_test, (c_test + np.float64(EPSILON)))
    # (a_a, b_a, c_a), out_a = compute_f((a_test + np.float64(EPSILON)), b_test, c_test)
    # (a_b, b_b, c_b), out_b = compute_f(a_test, (b_test + np.float64(EPSILON)), c_test)
    # (a_c, b_c, c_c), out_c = compute_f(a_test, b_test, (c_test + np.float64(EPSILON)))
    change_a_0 = compute_f(a_test + np.array([EPSILON, 0]), b_test, c_test)[1].value - orig
    change_a_1 = compute_f(a_test + np.array([0, EPSILON]), b_test, c_test)[1].value - orig
    change_b_0 = compute_f(a_test, b_test + np.array([EPSILON, 0]), c_test)[1].value - orig
    change_b_1 = compute_f(a_test, b_test + np.array([0, EPSILON]), c_test)[1].value - orig
    change_c_0 = compute_f(a_test, b_test, c_test + np.array([EPSILON, 0]))[1].value - orig
    change_c_1 = compute_f(a_test, b_test, c_test + np.array([0, EPSILON]))[1].value - orig
    # (a_a, b_a, c_a), out_a = compute_f(np.float64(a_test + EPSILON), b_test, c_test)
    # out_a.compute_grad()
    # (a_b, b_b, c_b), out_b = compute_f(a_test, np.float64(b_test + EPSILON), c_test)
    # out_b.compute_grad()
    # (a_c, b_c, c_c), out_c = compute_f(a_test, b_test, np.float64(c_test + EPSILON))
    # out_c.compute_grad()
    # change_a = a_a.grad - a.grad
    # change_b = b_b.grad - b.grad
    # change_c = c_c.grad - c.grad
    # change_a = out_a.value - orig
    # change_b = out_b.value - orig
    # change_c = out_c.value - orig

    # Derivative is (change in output) / (change in input)
    a_grad_numerical[0] = change_a_0 / np.array([EPSILON])
    a_grad_numerical[1] = change_a_1 / np.array([EPSILON])
    b_grad_numerical[0] = change_b_0 / np.array([EPSILON])
    b_grad_numerical[1] = change_b_1 / np.array([EPSILON])
    c_grad_numerical[0] = change_c_0 / np.array([EPSILON])
    c_grad_numerical[1] = change_c_1 / np.array([EPSILON])
    # a_grad_numerical = a_a.grad
    # b_grad_numerical = b_b.grad
    # c_grad_numerical = c_c.grad
    # a_grad_numerical = change_a
    # b_grad_numerical = change_b
    # c_grad_numerical = change_c
    # a_grad_numerical = change_a / EPSILON
    # b_grad_numerical = change_b / EPSILON
    # c_grad_numerical = change_c / EPSILON

    ### END_SOLUTION 1h

    # Finally, this code checks the difference between the gradients computed by backprop
    # and the gradients computed numerically.
    # The max difference should be smaller than 1e-7
    print('a.grad:')
    print(f'    Computed numerically  : {a_grad_numerical}')
    print(f'    Computed with backprop: {a.grad}')
    print(f'    Max difference: {np.max(np.abs(a_grad_numerical - a.grad))}')
    print('b.grad:')
    print(f'    Computed numerically  : {b_grad_numerical}')
    print(f'    Computed with backprop: {b.grad}')
    print(f'    Max difference: {np.max(np.abs(b_grad_numerical - b.grad))}')
    print('c.grad:')
    print(f'    Computed numerically  : {c_grad_numerical}')
    print(f'    Computed with backprop: {c.grad}')
    print(f'    Max difference: {np.max(np.abs(c_grad_numerical - c.grad))}')


def make_logistic_regression(dataset):
    """Create a function that computes the logistic regression loss.

    Args:
        dataset: A list of (x, y) pairs where x is a numpy array and y is in {-1, 1}
    Returns:
        compute_loss: A function that takes in numpy values w_val and b_val, and returns:
            - A tuple (w, b) of InputNode objects for w and b, respectively
            - total_loss, a Node whose value is the logistic regression loss for w and b on the given dataset
    """
    def compute_loss(w_val, b_val):
        topo_order = []
        w = InputNode(w_val, topo_order)
        b = InputNode(b_val, topo_order)
        # TODO: Set total_loss to be a Node whose value is the logistic regression loss
        total_loss = None
        ### BEGIN_SOLUTION 1i

        input_nodes = []  # Array of input nodes base on every data in dataset

        first_iteration = True
        for data in dataset:
            # Each InputNode will have a value of a numpy array of length 2
            input_nodes.append(InputNode(data[0], topo_order))

        for index in range(len(dataset)):
            w_dot_x = DotNode(w, input_nodes[index])

            w_dot_x_add_b = AddNode(w_dot_x, b)

            current_loss = ConstantMulNode(w_dot_x_add_b, dataset[index][1])
            logsig_current_loss = LogNode(SigmoidNode(current_loss))


            if first_iteration:
                # The example-specific output node will simply be assigned to the total_loss node:
                total_loss = ConstantMulNode(logsig_current_loss, -1)
                first_iteration = False
            else:
                # The example-specific output node will have to be added to the cumulative tally of total_loss node:
                total_loss = AddNode(total_loss, ConstantMulNode(logsig_current_loss, -1))

        ### END_SOLUTION 1i
        return (w, b), total_loss

    return compute_loss

def gradient_descent(f, params, lr, num_iters):
    """Run gradient descent on a given function.

    Args:
        - f: A function that takes in a list of parameter values and returns
            - param_nodes, an iterable of InputNode objects for each parameter
            - loss, a Node object whose value is the loss being optimized
        - params: A list of initial parameter values as numpy objects.
            This list will get updated with the values chosen by gradient descent
        - lr: Learning rate for gradient descent
        - num_iters: Number of iterations for gradient descent
    """
    for t in range(num_iters):
        input_nodes, loss = f(*params)  # Run the loss function f with the current parameters
        # input_nodes is a tuple containing the InputNode for each parameter in params
        if t % 100 == 0:
            print(f't={t}: loss={loss.value}')

        # TODO: Update each parameter in params via gradient descent
        ### BEGIN_SOLUTION 1i

        loss.compute_grad()
        for index in range(len(params)):
            params[index] -= lr * input_nodes[index].grad
            # params[index] = param - lr * loss.grad
        ### END_SOLUTION 1i

    print(f'Final loss={loss.value}')

def gradient_check_2(M_test, v_test):
    """Run gradient check for MVMulNode."""
    # First let's define a helper function.
    # Given numpy arrays M_val and v_val, this function:
    #     - Creates a computation graph that computes f(M_val, v_val)
    #     - Returns the output node of this computation graph
    def compute_f(M_val, v_val):
        """Helper function for gradient checking.

        Args: numpy array values for M and c
        Returns: 2-tuple containing:
            - input_nodes: list of InputNode objects corresponding to M and v
            - output_node: The output node whose value is f(M, v)
        """
        # Create the shared topo_order list and InputNode's
        topo_order = []
        M = InputNode(M_val, topo_order)
        v = InputNode(v_val, topo_order)
        input_nodes = [M, v]

        # Now, create the computation graph that computes f
        output_node = None  # TODO: Set this so we can return it
        ### BEGIN_SOLUTION 1m
        M_dot_v = MVMulNode(M, v)
        relu_M_dot_v = ReluNode(M_dot_v)
        output_node = DotNode(v, relu_M_dot_v)
        ### END_SOLUTION 1m
        return input_nodes, output_node

    # Compute gradients with backpropagation
    (M, v), out = compute_f(M_test, v_test)
    print(f'Computed value of f(M_test, v_test): {out.value}')
    out.compute_grad()  # Runs backpropagation, setting M.grad and v.grad

    # Now, numerically compute the gradient by changing each coordinate of each input by EPSILON
    # and observing how much f changes.
    # TODO: store the numerical gradients in the variables below.
    M_grad_numerical = np.zeros((2, 2))  # Numerically computed gradient with respect to M
    v_grad_numerical = np.zeros(2)  # Numerically computed gradient with respect to v
    ### BEGIN_SOLUTION 1m
    orig = out.value
    change_M_0_0 = compute_f(M_test + np.array([[EPSILON, 0], [0, 0]]), v_test)[1].value - orig
    change_M_0_1 = compute_f(M_test + np.array([[0, EPSILON], [0, 0]]), v_test)[1].value - orig
    change_M_1_0 = compute_f(M_test + np.array([[0, 0], [EPSILON, 0]]), v_test)[1].value - orig
    change_M_1_1 = compute_f(M_test + np.array([[0, 0], [0, EPSILON]]), v_test)[1].value - orig
    change_v_0 = compute_f(M_test, v_test + np.array([EPSILON, 0]))[1].value - orig
    change_v_1 = compute_f(M_test, v_test + np.array([0, EPSILON]))[1].value - orig

    M_grad_numerical[0][0] = change_M_0_0 / np.array([EPSILON])
    M_grad_numerical[0][1] = change_M_0_1 / np.array([EPSILON])
    M_grad_numerical[1][0] = change_M_1_0 / np.array([EPSILON])
    M_grad_numerical[1][1] = change_M_1_1 / np.array([EPSILON])
    v_grad_numerical[0] = change_v_0 / np.array([EPSILON])
    v_grad_numerical[1] = change_v_1 / np.array([EPSILON])
    ### END_SOLUTION 1m

    # Finally, this code checks the difference between the gradients computed by backprop
    # and the gradients computed numerically.
    # The max difference should be smaller than 1e-7
    print('M.grad:')
    print(f'    Computed numerically  : {M_grad_numerical}')
    print(f'    Computed with backprop: {M.grad}')
    print(f'    Max difference: {np.max(np.abs(M_grad_numerical - M.grad))}')
    print('v.grad:')
    print(f'    Computed numerically  : {v_grad_numerical}')
    print(f'    Computed with backprop: {v.grad}')
    print(f'    Max difference: {np.max(np.abs(v_grad_numerical - v.grad))}')

def make_neural_network(dataset):
    def compute_loss(W_val, b_val, v_val, c_val):
        topo_order = []
        W = InputNode(W_val, topo_order)
        b = InputNode(b_val, topo_order)
        v = InputNode(v_val, topo_order)
        c = InputNode(c_val, topo_order)
        # TODO: Set total_loss to be a Node whose value is the loss for the neural network
        total_loss = None
        ### BEGIN_SOLUTION 1n

        input_nodes = []  # Array of input nodes base on every data in dataset

        first_iteration = True
        for data in dataset:
            # Each InputNode will have a value of a numpy array of length 2
            input_nodes.append(InputNode(data[0], topo_order))

        for index in range(len(dataset)):
            w_dot_x = MVMulNode(W, input_nodes[index])
            w_dot_x_add_b = AddNode(w_dot_x, b)
            relu_w_dot_x_add_b = ReluNode(w_dot_x_add_b)
            v_dot_relu_w_dot_x_add_b = DotNode(v, relu_w_dot_x_add_b)
            v_dot_relu_w_dot_x_add_b_add_c = AddNode(v_dot_relu_w_dot_x_add_b, c)
            current_loss = ConstantMulNode(v_dot_relu_w_dot_x_add_b_add_c, dataset[index][1])
            logsig_current_loss = LogNode(SigmoidNode(current_loss))

            if first_iteration:
                # The example-specific output node will simply be assigned to the total_loss node:
                total_loss = ConstantMulNode(logsig_current_loss, -1)
                first_iteration = False
            else:
                # The example-specific output node will have to be added to the cumulative tally of total_loss node:
                total_loss = AddNode(total_loss, ConstantMulNode(logsig_current_loss, -1))

        ### END_SOLUTION 1n
        return (W, b, v, c), total_loss

    return compute_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['grad_check_1', 'grad_check_2', 'logreg', 'neural'])
    parser.add_argument('--dataset', '-d', choices=['simple', 'xor'])
    return parser.parse_args()


def main():
    if OPTS.mode == 'grad_check_1':
        a_test = np.array([3.0, 2.0])
        b_test = np.array([-2.0, 7.0])
        c_test = np.array([1.0, -4.0])
        gradient_check_1(a_test, b_test, c_test)
    if OPTS.mode == 'grad_check_2':
        M_test = np.array([[2.0, -1.0], [0.0, 3.0]])
        v_test = np.array([1.0, -3.0])
        gradient_check_2(M_test, v_test)
    else:
        if OPTS.dataset == 'simple':
            dataset = SIMPLE_DATASET
        elif OPTS.dataset == 'xor':
            dataset = XOR_DATASET
        if OPTS.mode == 'logreg':
            loss_func = make_logistic_regression(dataset)
            # Initial values of w and b
            params = [np.array([0.0, 0.0]), np.float64(0)]
            gradient_descent(loss_func, params, 1.0, 1000)
            print(f'Learned w: {params[0]}')
            print(f'Learned b: {params[1]}')
        elif OPTS.mode == 'neural':
            loss_func = make_neural_network(dataset)
            # Initial values of W, b, v, and c
            # We initialize with small random numbers to break symmetry
            params = [np.array([[-.01, .07], [.17, 0], [-.04, -.11]]),
                      np.array([-.02, -.14, .05]),
                      np.array([.05, .14, .12]),
                      np.float64(-.02)]
            gradient_descent(loss_func, params, 0.1, 1000)
            print(f'Learned W: {params[0]}')
            print(f'Learned b: {params[1]}')
            print(f'Learned v: {params[2]}')
            print(f'Learned c: {params[3]}')

if __name__ == '__main__':
    OPTS = parse_args()
    main()

