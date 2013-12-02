import math
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression as LR
import pprint as pp


def plot_sin(length=100):
	#clear the plot, we'll be plotting each random hypothesis
	plt.clf()

	#get the sin curve
	x = np.linspace(0,2*math.pi,length)
	y = [math.sin(n) for n in x]

	#initialize result set
	all_rand = np.zeros(length)
	all_line = np.zeros((length,length))

	mse_rand = np.zeros(length)
	mse_line = np.zeros(length)

	rand_bias= np.zeros(length)
	line_bias = np.zeros(length)
	
	for i in xrange(length):
		#get a random number between [-1,1] for rand hypothesis, i.e. initialize rand model
		rand = random.uniform(-1,1)
		rand1 = random.uniform(-10,10)

		#get the vector of the rand for plotting
		randvec = [rand for l in y]

		#initialize line model
		x1 = random.randint(0,len(y)-1)
		
		#this conditional prevents xa, from beint zero - which in turn 
		#prevents b from being nan or inf and breaking the model
		xq = x[x1]
		xa = 0
		if xq == 0:
			xa = 0.0001
		else:
			xa = xq

		b = (y[x1] - rand1)/xa

		#get the vector of the line model for plotting
		line_pred = [ (b*q + rand1) for q in x]

		#get the error of each hypothesis (randvec & line_pred) with the real sin curve
		mse_rand[i] = mse(y,randvec)
		mse_line[i] = mse(y,line_pred)

		#plot the hypothesis
		plt.plot(x,randvec)
		plt.plot(x,line_pred)

		#put the hypothesis in a vector
		all_rand[i] = rand
		all_line[i,:] = line_pred

		#find bias for both models
		b1 = [randvec[p] - y[p] for p in xrange(len(y))]
		rand_bias[i] = np.mean(b1)

		b2 = [line_pred[o] - y[o] for o in xrange(len(y))]
		line_bias[i] = np.mean(b2)



	#run result statistics for random model
	bias_rand_model = np.mean(rand_bias)
	rand_avg = np.mean(all_rand)
	MSE_rand = np.mean(mse_rand)
	Max_mse_rand = mse_rand.max()
	Min_mse_rand = mse_rand.min()
	std_mse_rand = np.std(mse_rand)
	rand_var = np.var(all_rand)
	#bias_random = (MSE_rand - rand_var)**(.5)

	#print rand model info
	print 'RANDOM MODEL'
	print '\nRand Model: Model bias =', bias_rand_model
	print 'Rand Model: Average hpyothesis =', rand_avg, '\n'

	print "In this case the average of the random model and the bias are the same because the random model"
	print "is a constant number, thus the mean of the model is going to be the same as its bias becuase the"
	print "mean of a constant is its expected value, i.e. its trend - this is equivalent to a model's bias" 
	print "in the constant number case\n" 

	print 'Rand Model: Average MSE across {0} trials = {1}'.format(length,MSE_rand)
	print 'Rand Model: Max MSE =', Max_mse_rand
	print 'Rand Model: Min MSE =', Min_mse_rand
	#print 'Rand Model: MSE Standard Deviation =', std_mse_rand
	print '\nRand Model: Variance =', rand_var
	#print 'another bias calc =', bias_random

	print '\n', '*+'*40, '\n'

	#run result statistics for line model
	bias_line_model = np.mean(line_bias)
	line_avg = [np.mean(i) for i in all_line]
	MSE_line = np.mean(mse_line)
	Max_mse_line = mse_line.max()
	Min_mse_line = mse_line.min()
	std_mse_line = np.std(mse_line)
	line_var = np.var(all_line)
	#bias_line = (MSE_line - line_var)**(.5)

	#print line model info
	print 'LINE MODEL\n'
	print 'Line Model: Model bias =', bias_line_model
	print 'Line Model: Average hypothesis =', np.mean(line_avg), '\n'

	print 'Line Model: Average MSE across {0} trials = {1}'.format(length,MSE_line)
	print 'Line Model: Max MSE =', Max_mse_line
	print 'Line Model: Min MSE =', Min_mse_line
	#print 'Line Model: MSE Standard Deviation =', std_mse_rand
	print '\nLine Model: Variance =', line_var
	#print 'another bias calc =', bias_line

	print '='*80
	print '='*80


	
	plt.plot(x,y)
	plt.ylim(-1.5,1.5)
	plt.show()