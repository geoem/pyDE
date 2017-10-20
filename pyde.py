from differential_evolution import *
from problems import *

problem = G01()                              		# Problem definition
de = DifferentialEvolution(problem, 100, 0.6, 0.9)  # Create DifferentialEvolution object
de.solve()                                          # Solve the problem

# Summary results
print "\nOptimization results using %s optimizer for problem %s" %(de.__class__.__name__, problem.__class__.__name__)
print "\n The solution x* found after %d function evaluations is:\n" % de.function_evaluations
for i in range(de.dimension):
	print " x%s = %f" % (i + 1, de.best.tolist()[i])
print "\n with minimum value f(x*): ", de.fitness_min
