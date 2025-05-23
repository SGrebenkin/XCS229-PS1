# Important note: you do not have to modify this file for your homework.
import math

import util
import numpy as np
import matplotlib.pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad


def logistic_regression(X, Y, data_set_name):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1
    norms = []

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad

        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            print('Theta: %s' % theta)

            fig = plt.figure(figsize=(12, 5))

            # 3D plot of data and regression surface
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(X[:, 1][Y == 0], X[:, 2][Y == 0], Y[Y == 0], c='green', label='y = 0.0')
            ax.scatter(X[:, 1][Y == 1], X[:, 2][Y == 1], Y[Y == 1], c='blue', label='y = 1.0')
            x0_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
            x1_range = np.linspace(X[:, 2].min(), X[:, 2].max(), 20)
            x0_grid, x1_grid = np.meshgrid(x0_range, x1_range)
            z_grid = 1 / (1 + np.exp(-(theta[0] + theta[1] * x0_grid + theta[2] * x1_grid)))
            ax.plot_surface(x0_grid, x1_grid, z_grid, alpha=0.3, color='red', edgecolor='k', linewidth=0.5, rstride=1,
                            cstride=1, label='Regression surface')
            ax.set_xlabel('x0')
            ax.set_ylabel('x1')
            ax.set_zlabel('y')
            ax.set_title('3D Visualization of Data and Regression Surface' + data_set_name)
            ax.legend()

            # Plot norm of parameter change vs iteration
            if 'norms' not in locals():
                norms = []
            norms.append(math.log(np.linalg.norm(prev_theta - theta)))
            ax2 = fig.add_subplot(122)
            ax2.plot(range(len(norms)), norms, marker='o')
            ax2.set_xlabel('Iteration (x10000)')
            ax2.set_ylabel('log(np.linalg.norm(prev_theta - theta))')
            ax2.set_title('Log(Norm) vs Iteration')

            plt.tight_layout()
            plt.show()
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya, " A dataset")

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb, " B dataset")


if __name__ == '__main__':
    main()
