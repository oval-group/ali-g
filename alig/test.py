import tensorflow as tf
import torch
import numpy as np
import unittest

from alig.tf import AliG as AliG_tf
from alig.th import AliG as AliG_th


class ModelTh(torch.nn.Module):
    def __init__(self, w1, b1, w2, b2):
        super(ModelTh, self).__init__()
        self.linear1 = torch.nn.Linear(w1.shape[0], w1.shape[1], bias=True)
        self.linear2 = torch.nn.Linear(w2.shape[0], w2.shape[1], bias=True)
        self.relu = torch.nn.ReLU()
        self.loss = torch.nn.CrossEntropyLoss()
        self._init(w1, b1, w2, b2)

    def _init(self, w1, b1, w2, b2):
        self.linear1.weight.data.copy_(torch.from_numpy(w1).t())
        self.linear1.bias.data.copy_(torch.from_numpy(b1))
        self.linear2.weight.data.copy_(torch.from_numpy(w2).t())
        self.linear2.bias.data.copy_(torch.from_numpy(b2))

    def forward(self, x, y):
        o = self.linear1(x)
        o = self.relu(o)
        o = self.linear2(o)
        o = self.loss(o, y)
        return o

    def get_weights(self):
        w1 = self.linear1.weight.data.t().numpy()
        b1 = self.linear1.bias.data.numpy()
        w2 = self.linear2.weight.data.t().numpy()
        b2 = self.linear2.bias.data.numpy()
        return w1, b1, w2, b2


class ModelTf():
    def __init__(self, w1, b1, w2, b2):
        with tf.variable_scope("weights"):
            X = tf.placeholder("float64", [None, w1.shape[0]])
            Y = tf.placeholder("float64", [None, w2.shape[1]])
            O = tf.add(tf.matmul(X, tf.Variable(w1, name='W1')), tf.Variable(b1, name='B1'))
            O = tf.nn.relu(O)
            O = tf.add(tf.matmul(O, tf.Variable(w2, name='W2')), tf.Variable(b2, name='B2'))
            O = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=O, labels=Y))

        self.X = X
        self.Y = Y
        self.out = O

    def get_weights(self):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="weights")
        w1, b1, w2, b2 = variables[:4]
        return w1.eval(), b1.eval(), w2.eval(), b2.eval()


class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        torch.set_default_dtype(torch.float64)

        self.max_lr = 3
        self.momentum = 0.9
        self.iterations = 5

        self.n_samples = 10
        self.h1 = 5
        self.h2 = 8
        self.n_classes = 3

        self.x = np.random.normal(0, 1, size=(self.n_samples, self.h1))
        self.y = np.random.randint(0, self.n_classes, size=(self.n_samples,))

        self.w1 = np.random.normal(0, 1, size=(self.h1, self.h2))
        self.b1 = np.random.normal(0, 1, size=(self.h2,))
        self.w2 = np.random.normal(0, 1, size=(self.h2, self.n_classes))
        self.b2 = np.random.normal(0, 1, size=(self.n_classes,))

        self.setup_th()
        self.setup_tf()

    def setup_th(self):
        torch.set_default_dtype(torch.double)
        self.model_th = ModelTh(self.w1, self.b1, self.w2, self.b2)
        self.feed_dict_th = {'x': torch.from_numpy(self.x), 'y': torch.from_numpy(self.y)}

    def setup_tf(self):
        tf.reset_default_graph()
        self.model_tf = ModelTf(self.w1, self.b1, self.w2, self.b2)
        y_one_hot = np.zeros((self.n_samples, self.n_classes), dtype=np.float32)
        y_one_hot[np.arange(0, self.n_samples), self.y] = 1
        self.feed_dict_tf = {self.model_tf.X: self.x, self.model_tf.Y: y_one_hot}

    def check_model_weights(self, it, atol=1e-5, rtol=1e-5):
        state_tf = self.model_tf.get_weights()
        state_th = self.model_th.get_weights()
        for w1, w2 in zip(state_tf, state_th):
            assert np.allclose(w1, w2, atol=atol, rtol=rtol), "Failed check at iteration {}".format(it)

    def test_alig_without_momentum(self):
        optimizer_tf = AliG_tf()
        optimizer_th = AliG_th(self.model_th.parameters())

        self.train_with(optimizer_tf, optimizer_th)

    def test_alig_with_momentum(self):
        optimizer_tf = AliG_tf(momentum=self.momentum)
        optimizer_th = AliG_th(self.model_th.parameters(),
                               momentum=self.momentum, adjusted_momentum=True)

        self.train_with(optimizer_tf, optimizer_th)

    def test_alig_max_lr_without_momentum(self):
        optimizer_tf = AliG_tf(max_lr=self.max_lr)
        optimizer_th = AliG_th(self.model_th.parameters(), max_lr=self.max_lr)

        self.train_with(optimizer_tf, optimizer_th)

    def test_alig_max_lr_with_momentum(self):
        optimizer_tf = AliG_tf(max_lr=self.max_lr, momentum=self.momentum)
        optimizer_th = AliG_th(self.model_th.parameters(), max_lr=self.max_lr,
                               momentum=self.momentum, adjusted_momentum=True)

        self.train_with(optimizer_tf, optimizer_th)

    def train_with(self, optimizer_tf, optimizer_th):
        train_op = optimizer_tf.minimize(self.model_tf.out)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            # check initial weights
            self.check_model_weights(it=0)

            # iteration loop
            for i in range(self.iterations):
                # tf iteration
                _, loss_tf = sess.run((train_op, self.model_tf.out),
                                      feed_dict=self.feed_dict_tf)

                # th iteration
                loss_th = self.model_th(**self.feed_dict_th)
                optimizer_th.zero_grad()
                loss_th.backward()
                optimizer_th.step(lambda: loss_th)

                # check weights
                self.check_model_weights(it=i + 1)


if __name__ == '__main__':
    unittest.main()
